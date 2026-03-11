"""
Causal Discovery via DAG — Descubrir la ESTRUCTURA causal detras de anomalias.

En lugar de correlaciones, descubrimos causalidad:
  - Infraestructura vieja CAUSA anomalias?
  - Poblacion mayor CAUSA patrones de consumo diferentes?

Implementa el algoritmo PC (constraint-based) desde cero usando
tests de independencia condicional (correlacion parcial).

Uso:
  from causal_analysis import run_causal_analysis, causal_summary
  results = run_causal_analysis(df)
  causal_summary(results)

Solo usa: pandas, numpy, scipy.stats. SIN librerias causales externas.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

__all__ = ["run_causal_analysis", "causal_summary"]

# ─────────────────────────────────────────────────────────────────
# Variables del DAG
# ─────────────────────────────────────────────────────────────────

DAG_VARIABLES = [
    "pct_elderly_65plus",
    "population_density",
    "anr_ratio",
    "consumption_per_contract",
    "deviation_from_group_trend",
    "anomaly_score",
    "n_models_detecting",
]

# Variables exogenas: no pueden tener padres (causas entrantes)
EXOGENOUS = {"pct_elderly_65plus", "population_density"}

# Variables de deteccion: solo pueden ser efectos finales
DETECTION_OUTPUTS = {"anomaly_score", "n_models_detecting"}

# Etiquetas legibles
LABELS = {
    "pct_elderly_65plus": "Demografia (>65)",
    "population_density": "Densidad poblacional",
    "anr_ratio": "ANR (perdida fisica)",
    "consumption_per_contract": "Consumo/contrato",
    "deviation_from_group_trend": "Desviacion de tendencia",
    "anomaly_score": "Anomaly Score",
    "n_models_detecting": "N modelos detectando",
}


# ─────────────────────────────────────────────────────────────────
# Utilidades: correlacion parcial y tests de independencia
# ─────────────────────────────────────────────────────────────────

def _standardize(df, cols):
    """Estandarizar columnas a media 0, std 1."""
    result = df[cols].copy()
    for c in cols:
        mu = result[c].mean()
        sd = result[c].std()
        if sd > 1e-10:
            result[c] = (result[c] - mu) / sd
        else:
            result[c] = 0.0
    return result


def _partial_correlation(x, y, z_cols, data):
    """
    Correlacion parcial entre x e y controlando por z_cols.

    Usa regresion lineal: residuos de x ~ z y residuos de y ~ z,
    luego correlacion entre residuos.

    Returns: (partial_corr, p_value)
    """
    if len(z_cols) == 0:
        # Sin control: correlacion simple
        valid = data[[x, y]].dropna()
        if len(valid) < 5:
            return 0.0, 1.0
        r, p = stats.pearsonr(valid[x], valid[y])
        return r, p

    cols_needed = [x, y] + list(z_cols)
    valid = data[cols_needed].dropna()
    n = len(valid)

    if n < len(z_cols) + 5:
        return 0.0, 1.0

    Z = valid[list(z_cols)].values
    X = valid[x].values
    Y = valid[y].values

    # Residuos de X ~ Z
    Z_pinv = np.linalg.pinv(Z)
    beta_x = Z_pinv @ X
    resid_x = X - Z @ beta_x

    # Residuos de Y ~ Z
    beta_y = Z_pinv @ Y
    resid_y = Y - Z @ beta_y

    # Correlacion entre residuos
    sx = np.std(resid_x)
    sy = np.std(resid_y)
    if sx < 1e-10 or sy < 1e-10:
        return 0.0, 1.0

    r = np.corrcoef(resid_x, resid_y)[0, 1]

    # Test de significancia: Fisher z-transform
    k = len(z_cols)
    dof = n - k - 2
    if dof < 2:
        return r, 1.0

    # t-statistic from partial correlation
    t_stat = r * np.sqrt(dof / (1.0 - r**2 + 1e-15))
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=dof)

    return r, p_value


# ─────────────────────────────────────────────────────────────────
# 1. DISCOVER CAUSAL STRUCTURE (algoritmo PC)
# ─────────────────────────────────────────────────────────────────

def discover_causal_structure(results_df):
    """
    Implementa el algoritmo PC (Peter-Clark) para descubrir la estructura
    causal a partir de tests de independencia condicional.

    Pasos:
      1. Empezar con grafo completo no dirigido
      2. Eliminar aristas via tests de independencia condicional (correlacion parcial)
      3. Orientar aristas usando v-structures (colliders)
      4. Aplicar restricciones de dominio (exogenas, outputs)

    Args:
        results_df: DataFrame con columnas de DAG_VARIABLES

    Returns:
        dict con:
          - adjacency: matriz NxN (1 = arista X->Y)
          - edges: lista de (source, target, info)
          - skeleton: grafo no dirigido antes de orientacion
          - separation_sets: conjuntos de separacion por par
          - variables: lista de nombres
    """
    # Preparar datos: agregar por barrio (usar ultima observacion o media)
    if "barrio_key" in results_df.columns:
        # Agrupar por barrio para tener una fila por unidad
        agg_funcs = {}
        for v in DAG_VARIABLES:
            if v in results_df.columns:
                agg_funcs[v] = "mean"
        df = results_df.groupby("barrio_key").agg(agg_funcs).reset_index()
    else:
        df = results_df.copy()

    # Verificar que las variables existen
    available = [v for v in DAG_VARIABLES if v in df.columns]
    missing = [v for v in DAG_VARIABLES if v not in df.columns]
    if missing:
        print(f"  [AVISO] Variables no disponibles: {missing}")

    if len(available) < 3:
        print("  [ERROR] Menos de 3 variables disponibles para DAG")
        return {
            "adjacency": np.zeros((0, 0)),
            "edges": [],
            "skeleton": np.zeros((0, 0)),
            "separation_sets": {},
            "variables": [],
        }

    n_vars = len(available)
    var_idx = {v: i for i, v in enumerate(available)}

    # Estandarizar datos
    data = _standardize(df, available)
    data = data.dropna()
    n_obs = len(data)
    print(f"\n  [DAG] Algoritmo PC con {n_vars} variables, {n_obs} observaciones")

    # ── Paso 1: Grafo completo no dirigido ──
    # skeleton[i,j] = 1 significa arista entre i y j (no dirigida)
    skeleton = np.ones((n_vars, n_vars), dtype=int)
    np.fill_diagonal(skeleton, 0)

    # Conjuntos de separacion: sep_sets[(i,j)] = conjunto que hace i _||_ j
    sep_sets = {}

    # Nivel de significancia para tests
    alpha = 0.05

    # ── Paso 2: Eliminar aristas por independencia condicional ──
    # Iterar sobre tamanios de conditioning set: 0, 1, 2, ...
    max_cond_size = min(n_vars - 2, 4)  # Limitar para estabilidad

    for cond_size in range(0, max_cond_size + 1):
        # Para cada par con arista
        edges_to_check = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] == 1:
                    edges_to_check.append((i, j))

        for i, j in edges_to_check:
            if skeleton[i, j] == 0:
                continue  # Ya eliminada en esta iteracion

            # Vecinos de i (excluyendo j)
            neighbors_i = [k for k in range(n_vars)
                           if k != i and k != j and skeleton[i, k] == 1]

            # Si no hay suficientes vecinos para este cond_size, skip
            if len(neighbors_i) < cond_size:
                continue

            # Probar todos los subconjuntos de tamanio cond_size
            found_independent = False
            for cond_set in combinations(neighbors_i, cond_size):
                z_cols = [available[k] for k in cond_set]
                r, p = _partial_correlation(
                    available[i], available[j], z_cols, data
                )

                if p > alpha:
                    # i y j son condicionalmente independientes dado cond_set
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    sep_sets[(i, j)] = set(cond_set)
                    sep_sets[(j, i)] = set(cond_set)
                    found_independent = True
                    break

            if found_independent:
                continue

            # Tambien probar vecinos de j
            neighbors_j = [k for k in range(n_vars)
                           if k != i and k != j and skeleton[j, k] == 1]

            if len(neighbors_j) < cond_size:
                continue

            for cond_set in combinations(neighbors_j, cond_size):
                z_cols = [available[k] for k in cond_set]
                r, p = _partial_correlation(
                    available[i], available[j], z_cols, data
                )

                if p > alpha:
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    sep_sets[(i, j)] = set(cond_set)
                    sep_sets[(j, i)] = set(cond_set)
                    break

    print(f"  [DAG] Esqueleto: {skeleton.sum() // 2} aristas no dirigidas")

    # ── Paso 3: Orientar aristas usando v-structures (colliders) ──
    # Para cada triple i - k - j donde i y j NO estan conectados:
    #   si k NO esta en sep_set(i,j), orientar como i -> k <- j (collider)

    # adjacency[i,j] = 1 significa arista dirigida i -> j
    adjacency = np.zeros((n_vars, n_vars), dtype=int)

    # Copiar skeleton como base (ambas direcciones posibles)
    oriented = set()  # Aristas ya orientadas

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton[i, j] == 0:
                continue
            # Buscar v-structures: i - k - j con i no adyacente a j... wait
            # Aqui skeleton[i,j] == 1, asi que estan conectados
            pass

    # V-structures: para cada par NO conectado (i, j), buscar k comun
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton[i, j] == 1:
                continue  # Estan conectados, no aplica

            # Buscar nodos k adyacentes a AMBOS i y j
            for k in range(n_vars):
                if k == i or k == j:
                    continue
                if skeleton[i, k] == 0 or skeleton[j, k] == 0:
                    continue

                # k es adyacente a i y a j, pero i y j no son adyacentes
                # Verificar si k esta en el separation set de (i, j)
                s = sep_sets.get((i, j), set())
                if k not in s:
                    # V-structure: i -> k <- j
                    adjacency[i, k] = 1
                    adjacency[j, k] = 1
                    oriented.add((i, k))
                    oriented.add((j, k))

    # ── Paso 4: Orientar aristas restantes por reglas de Meek + dominio ──
    # Regla de dominio: variables exogenas no pueden tener padres
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            if skeleton[i, j] == 0 and skeleton[j, i] == 0:
                # Si no hay arista en el skeleton original, no hay arista en DAG
                if adjacency[i, j] == 0:
                    continue

            vi = available[i]
            vj = available[j]

            # Exogenas: no pueden tener padres -> solo salen aristas
            if vj in EXOGENOUS and (j, i) not in oriented:
                # j es exogena, no puede recibir arista de i
                adjacency[i, j] = 0
                if skeleton[i, j] == 1 and (i, j) not in oriented:
                    adjacency[j, i] = 1
                    oriented.add((j, i))

            # Outputs de deteccion: no causan variables de entrada
            if vi in DETECTION_OUTPUTS and vj not in DETECTION_OUTPUTS:
                adjacency[i, j] = 0
                if skeleton[i, j] == 1 and (j, i) not in oriented:
                    adjacency[j, i] = 1
                    oriented.add((j, i))

    # Orientar aristas restantes del skeleton que no se han orientado
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if skeleton[i, j] == 0:
                continue
            if adjacency[i, j] == 0 and adjacency[j, i] == 0:
                # Arista no orientada: usar correlacion con anomaly_score
                # como heuristica + restricciones de dominio
                vi = available[i]
                vj = available[j]

                # Heuristica: variable mas "upstream" causa la otra
                # Orden causal plausible por dominio
                causal_order = {
                    "pct_elderly_65plus": 0,
                    "population_density": 0,
                    "anr_ratio": 1,
                    "consumption_per_contract": 2,
                    "deviation_from_group_trend": 3,
                    "anomaly_score": 4,
                    "n_models_detecting": 5,
                }
                oi = causal_order.get(vi, 3)
                oj = causal_order.get(vj, 3)

                if oi < oj:
                    adjacency[i, j] = 1
                    oriented.add((i, j))
                elif oj < oi:
                    adjacency[j, i] = 1
                    oriented.add((j, i))
                else:
                    # Mismo orden: usar fuerza de correlacion con anomaly_score
                    if "anomaly_score" in var_idx:
                        a_idx = var_idx["anomaly_score"]
                        ri, _ = _partial_correlation(vi, "anomaly_score", [], data)
                        rj, _ = _partial_correlation(vj, "anomaly_score", [], data)
                        if abs(ri) < abs(rj):
                            adjacency[i, j] = 1
                        else:
                            adjacency[j, i] = 1
                    else:
                        adjacency[i, j] = 1

    # Eliminar ciclos: si i->j y j->i, mantener solo la direccion con
    # mayor correlacion parcial
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if adjacency[i, j] == 1 and adjacency[j, i] == 1:
                r_ij, _ = _partial_correlation(available[i], available[j], [], data)
                # Mantener i->j si i tiene menor orden causal
                causal_order = {
                    "pct_elderly_65plus": 0, "population_density": 0,
                    "anr_ratio": 1, "consumption_per_contract": 2,
                    "deviation_from_group_trend": 3,
                    "anomaly_score": 4, "n_models_detecting": 5,
                }
                oi = causal_order.get(available[i], 3)
                oj = causal_order.get(available[j], 3)
                if oi <= oj:
                    adjacency[j, i] = 0
                else:
                    adjacency[i, j] = 0

    # Construir lista de aristas
    edges = []
    for i in range(n_vars):
        for j in range(n_vars):
            if adjacency[i, j] == 1:
                r, p = _partial_correlation(available[i], available[j], [], data)
                edges.append({
                    "source": available[i],
                    "target": available[j],
                    "source_label": LABELS.get(available[i], available[i]),
                    "target_label": LABELS.get(available[j], available[j]),
                    "correlation": r,
                    "p_value": p,
                })

    print(f"  [DAG] Grafo dirigido: {len(edges)} aristas orientadas")

    return {
        "adjacency": adjacency,
        "edges": edges,
        "skeleton": skeleton,
        "separation_sets": {str(k): list(v) for k, v in sep_sets.items()},
        "variables": available,
        "data": data,
    }


# ─────────────────────────────────────────────────────────────────
# 2. ESTIMATE CAUSAL EFFECTS (back-door adjustment)
# ─────────────────────────────────────────────────────────────────

def _get_parents(adjacency, variables, node_idx):
    """Obtener indices de padres de un nodo en el DAG."""
    return [i for i in range(len(variables)) if adjacency[i, node_idx] == 1]


def _find_all_directed_paths(adjacency, source, target, n_vars):
    """
    Encontrar todos los caminos dirigidos de source a target en el DAG.
    Retorna lista de listas de indices.
    """
    paths = []
    stack = [(source, [source])]

    while stack:
        current, path = stack.pop()
        if current == target and len(path) > 1:
            paths.append(path)
            continue
        for j in range(n_vars):
            if adjacency[current, j] == 1 and j not in path:
                stack.append((j, path + [j]))

    return paths


def estimate_causal_effects(results_df, dag):
    """
    Estimar efectos causales para cada arista del DAG.

    Para cada arista X -> Y:
      - Efecto directo: coeficiente beta de regresion Y ~ X + Parents(Y)
      - Efecto total sobre anomaly_score: producto de betas a lo largo
        de todos los caminos dirigidos

    Args:
        results_df: DataFrame original
        dag: dict retornado por discover_causal_structure

    Returns:
        pd.DataFrame con columnas:
          cause, effect, direct_beta, total_effect, p_value
    """
    adjacency = dag["adjacency"]
    variables = dag["variables"]
    data = dag["data"]
    n_vars = len(variables)

    if n_vars == 0:
        return pd.DataFrame(columns=[
            "cause", "effect", "direct_beta", "total_effect", "p_value"
        ])

    var_idx = {v: i for i, v in enumerate(variables)}

    # Calcular efecto directo para cada arista: regresion con back-door adjustment
    edge_betas = {}  # (i, j) -> (beta, p_value)

    for edge in dag["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        i = var_idx[src]
        j = var_idx[tgt]

        # Parents de j (excluyendo i): set de ajuste (back-door)
        parents_j = _get_parents(adjacency, variables, j)
        adjustment_set = [variables[p] for p in parents_j if p != i]

        # Regresion: tgt ~ src + adjustment_set
        predictors = [src] + adjustment_set
        valid = data[predictors + [tgt]].dropna()

        if len(valid) < len(predictors) + 3:
            edge_betas[(i, j)] = (0.0, 1.0)
            continue

        X = valid[predictors].values
        y = valid[tgt].values

        # OLS
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            edge_betas[(i, j)] = (0.0, 1.0)
            continue

        # beta del src es el segundo coeficiente (primero es intercept)
        direct_beta = beta_hat[1]

        # Calcular p-value del coeficiente
        y_pred = X_with_const @ beta_hat
        residuals = y - y_pred
        n = len(y)
        p = X_with_const.shape[1]
        dof = n - p

        if dof > 0:
            mse = np.sum(residuals**2) / dof
            try:
                var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                se_beta = np.sqrt(np.abs(var_beta[1, 1]))
                if se_beta > 1e-15:
                    t_stat = direct_beta / se_beta
                    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=dof)
                else:
                    p_value = 0.0
            except np.linalg.LinAlgError:
                p_value = 1.0
        else:
            p_value = 1.0

        edge_betas[(i, j)] = (direct_beta, p_value)

    # Calcular efecto total de cada variable sobre anomaly_score
    total_effects = {}

    if "anomaly_score" in var_idx:
        target_idx = var_idx["anomaly_score"]

        for src_var in variables:
            if src_var == "anomaly_score":
                continue
            src_idx = var_idx[src_var]
            paths = _find_all_directed_paths(adjacency, src_idx, target_idx, n_vars)

            total = 0.0
            for path in paths:
                # Producto de betas a lo largo del camino
                path_effect = 1.0
                for step in range(len(path) - 1):
                    beta, _ = edge_betas.get((path[step], path[step + 1]), (0.0, 1.0))
                    path_effect *= beta
                total += path_effect

            total_effects[src_var] = total

    # Construir DataFrame de resultados
    rows = []
    for edge in dag["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        i = var_idx[src]
        j = var_idx[tgt]
        beta, p_val = edge_betas.get((i, j), (0.0, 1.0))

        rows.append({
            "cause": src,
            "effect": tgt,
            "cause_label": LABELS.get(src, src),
            "effect_label": LABELS.get(tgt, tgt),
            "direct_beta": beta,
            "total_effect": total_effects.get(src, 0.0),
            "p_value": p_val,
            "significant": p_val < 0.05,
        })

    effects_df = pd.DataFrame(rows)
    if len(effects_df) > 0:
        effects_df = effects_df.sort_values("p_value").reset_index(drop=True)

    return effects_df


# ─────────────────────────────────────────────────────────────────
# 3. COMPUTE COUNTERFACTUAL IMPACT
# ─────────────────────────────────────────────────────────────────

def compute_counterfactual_impact(results_df, dag, effects):
    """
    Para barrios anomalos (anomaly_score > 0.3), estimar:
      - Si pct_elderly_65plus fuera la mediana de la ciudad,
        cuanto cambiaria anomaly_score?
      - Si anr_ratio fuera 1.0 (sin perdida), cuanto cambiaria?
      - Que fraccion de la anomalia se debe a cada factor?

    Args:
        results_df: DataFrame original
        dag: dict del DAG
        effects: DataFrame de efectos causales

    Returns:
        pd.DataFrame con impacto contrafactual por barrio y causa
    """
    variables = dag["variables"]

    if len(variables) == 0 or len(effects) == 0:
        return pd.DataFrame()

    # Preparar datos agregados por barrio
    if "barrio_key" in results_df.columns:
        agg_funcs = {}
        for v in variables:
            if v in results_df.columns:
                agg_funcs[v] = "mean"
        if "barrio" in results_df.columns:
            agg_funcs["barrio"] = "first"
        df = results_df.groupby("barrio_key").agg(agg_funcs).reset_index()
    else:
        df = results_df.copy()

    if "anomaly_score" not in df.columns:
        return pd.DataFrame()

    # Barrios anomalos
    anomalous = df[df["anomaly_score"] > 0.3].copy()
    if len(anomalous) == 0:
        # Relajar umbral
        anomalous = df.nlargest(10, "anomaly_score").copy()

    if len(anomalous) == 0:
        return pd.DataFrame()

    # Calcular efectos totales sobre anomaly_score desde effects
    total_effects = {}
    for _, row in effects.iterrows():
        cause = row["cause"]
        te = row["total_effect"]
        if te != 0 and cause != "anomaly_score":
            # Guardar el mayor efecto total por causa
            if cause not in total_effects or abs(te) > abs(total_effects[cause]):
                total_effects[cause] = te

    # Escenarios contrafactuales
    interventions = {}

    if "pct_elderly_65plus" in df.columns and "pct_elderly_65plus" in total_effects:
        city_median_elderly = df["pct_elderly_65plus"].median()
        interventions["pct_elderly_65plus"] = {
            "target_value": city_median_elderly,
            "label": "Si demografia (>65) = mediana ciudad",
            "category": "ESTRUCTURAL",
        }

    if "anr_ratio" in df.columns and "anr_ratio" in total_effects:
        interventions["anr_ratio"] = {
            "target_value": 1.0,
            "label": "Si ANR = 1.0 (sin perdida fisica)",
            "category": "ACCIONABLE",
        }

    if "deviation_from_group_trend" in df.columns and "deviation_from_group_trend" in total_effects:
        interventions["deviation_from_group_trend"] = {
            "target_value": 0.0,
            "label": "Si desviacion de tendencia = 0",
            "category": "ACCIONABLE",
        }

    if "population_density" in df.columns and "population_density" in total_effects:
        city_median_density = df["population_density"].median()
        interventions["population_density"] = {
            "target_value": city_median_density,
            "label": "Si densidad = mediana ciudad",
            "category": "ESTRUCTURAL",
        }

    if "consumption_per_contract" in df.columns and "consumption_per_contract" in total_effects:
        city_median_consumption = df["consumption_per_contract"].median()
        interventions["consumption_per_contract"] = {
            "target_value": city_median_consumption,
            "label": "Si consumo/contrato = mediana ciudad",
            "category": "ACCIONABLE",
        }

    # Estandarizar igual que en el DAG para usar los betas
    data_std = _standardize(df, [v for v in variables if v in df.columns])
    std_params = {}
    for v in variables:
        if v in df.columns:
            std_params[v] = {"mean": df[v].mean(), "std": df[v].std()}

    rows = []
    for _, barrio_row in anomalous.iterrows():
        barrio_name = barrio_row.get("barrio",
                                     barrio_row.get("barrio_key", "Desconocido"))
        barrio_key = barrio_row.get("barrio_key", barrio_name)
        current_anomaly = barrio_row["anomaly_score"]

        impacts = {}
        total_explained = 0.0

        for var, intervention in interventions.items():
            if var not in df.columns or var not in total_effects:
                continue

            current_val = barrio_row[var]
            target_val = intervention["target_value"]

            # Delta en unidades estandarizadas
            std = std_params[var]["std"]
            if std < 1e-10:
                continue

            delta_std = (target_val - current_val) / std

            # Cambio esperado en anomaly_score (estandarizado) via efecto total
            delta_anomaly_std = delta_std * total_effects[var]

            # Convertir a escala original de anomaly_score
            as_std = std_params.get("anomaly_score", {}).get("std", 1.0)
            delta_anomaly = delta_anomaly_std * as_std

            # Fraccion del anomaly_score explicada
            fraction = 0.0
            if current_anomaly > 1e-6:
                fraction = abs(delta_anomaly) / current_anomaly

            impacts[var] = {
                "delta": delta_anomaly,
                "fraction": fraction,
            }
            total_explained += abs(delta_anomaly)

        # Normalizar fracciones para que sumen <= 1
        if total_explained > 0:
            norm_factor = min(1.0, current_anomaly / total_explained)
        else:
            norm_factor = 0.0

        for var, intervention in interventions.items():
            if var not in impacts:
                continue

            impact = impacts[var]
            normalized_fraction = impact["fraction"] * norm_factor
            # Clamp entre 0 y 1
            normalized_fraction = max(0.0, min(1.0, normalized_fraction))

            rows.append({
                "barrio_key": barrio_key,
                "barrio": barrio_name,
                "anomaly_score": current_anomaly,
                "variable": var,
                "variable_label": LABELS.get(var, var),
                "intervention": intervention["label"],
                "category": intervention["category"],
                "current_value": barrio_row.get(var, np.nan),
                "counterfactual_value": intervention["target_value"],
                "delta_anomaly_score": impact["delta"],
                "fraction_explained": normalized_fraction,
                "total_effect": total_effects.get(var, 0.0),
            })

    cf_df = pd.DataFrame(rows)
    if len(cf_df) > 0:
        cf_df = cf_df.sort_values(
            ["anomaly_score", "fraction_explained"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return cf_df


# ─────────────────────────────────────────────────────────────────
# 4. RUN CAUSAL ANALYSIS (entry point)
# ─────────────────────────────────────────────────────────────────

def run_causal_analysis(results_df):
    """
    Punto de entrada principal: ejecuta todo el pipeline de analisis causal.

    1. Descubrir estructura causal (DAG) via algoritmo PC
    2. Estimar efectos causales (back-door adjustment)
    3. Calcular impacto contrafactual por barrio

    Args:
        results_df: DataFrame con resultados (de results_full.csv)

    Returns:
        dict con:
          - dag: estructura causal descubierta
          - effects: DataFrame de efectos causales
          - counterfactuals: DataFrame de impacto contrafactual
    """
    print(f"\n{'='*80}")
    print(f"  CAUSAL DISCOVERY — Estructura causal via DAG (algoritmo PC)")
    print(f"{'='*80}")

    # 1. Descubrir estructura
    dag = discover_causal_structure(results_df)

    # 2. Estimar efectos
    print(f"\n  [EFECTOS] Estimando efectos causales (back-door adjustment)...")
    effects = estimate_causal_effects(results_df, dag)
    if len(effects) > 0:
        sig = effects[effects["significant"]]
        print(f"  [EFECTOS] {len(sig)}/{len(effects)} aristas significativas (p<0.05)")

    # 3. Contrafactuales
    print(f"\n  [CONTRAFACTUAL] Calculando impacto por barrio...")
    counterfactuals = compute_counterfactual_impact(results_df, dag, effects)
    if len(counterfactuals) > 0:
        n_barrios = counterfactuals["barrio_key"].nunique()
        print(f"  [CONTRAFACTUAL] {n_barrios} barrios anomalos analizados")

    return {
        "dag": dag,
        "effects": effects,
        "counterfactuals": counterfactuals,
    }


# ─────────────────────────────────────────────────────────────────
# 5. CAUSAL SUMMARY (print)
# ─────────────────────────────────────────────────────────────────

def causal_summary(causal_results):
    """
    Imprimir resumen legible del analisis causal.

    Muestra:
      - DAG descubierto como texto (A -> B [beta=X.XX])
      - Top caminos causales a anomaly_score
      - Analisis contrafactual por barrio
      - Separacion de causas ACCIONABLES vs ESTRUCTURALES
    """
    dag = causal_results["dag"]
    effects = causal_results["effects"]
    counterfactuals = causal_results["counterfactuals"]

    print(f"\n{'='*80}")
    print(f"  RESUMEN DE ANALISIS CAUSAL")
    print(f"{'='*80}")

    # ── DAG descubierto ──
    print(f"\n  GRAFO CAUSAL DESCUBIERTO (DAG):")
    print(f"  {'─'*60}")

    if len(dag.get("edges", [])) == 0:
        print(f"    No se descubrieron aristas causales")
    else:
        for edge in dag["edges"]:
            src_label = LABELS.get(edge["source"], edge["source"])
            tgt_label = LABELS.get(edge["target"], edge["target"])
            r = edge["correlation"]
            p = edge["p_value"]
            sig = "*" if p < 0.05 else " "
            print(f"    {src_label:30s} -> {tgt_label:25s} "
                  f"[r={r:+.3f}, p={p:.3f}]{sig}")

    # ── Efectos causales ──
    if len(effects) > 0:
        print(f"\n  EFECTOS CAUSALES DIRECTOS (beta estandarizado):")
        print(f"  {'─'*60}")

        for _, row in effects.iterrows():
            sig_mark = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else ""))
            print(f"    {row['cause_label']:30s} -> {row['effect_label']:25s} "
                  f"beta={row['direct_beta']:+.3f} "
                  f"(p={row['p_value']:.3f}) {sig_mark}")

        # Efectos totales sobre anomaly_score
        total_on_anomaly = effects[
            effects["effect"] == "anomaly_score"
        ].copy() if "anomaly_score" in effects["effect"].values else pd.DataFrame()

        # Tambien incluir efectos indirectos
        causes_with_total = effects.drop_duplicates("cause").copy()
        causes_with_total = causes_with_total[
            causes_with_total["total_effect"].abs() > 1e-6
        ].sort_values("total_effect", key=abs, ascending=False)

        if len(causes_with_total) > 0:
            print(f"\n  EFECTO TOTAL SOBRE ANOMALY SCORE (via todos los caminos):")
            print(f"  {'─'*60}")
            for _, row in causes_with_total.iterrows():
                direction = "+" if row["total_effect"] > 0 else "-"
                bar_len = int(min(30, abs(row["total_effect"]) * 30))
                bar = "█" * max(1, bar_len)
                print(f"    {row['cause_label']:30s} {direction} {bar:30s} "
                      f"{row['total_effect']:+.3f}")

    # ── Top caminos causales ──
    if len(effects) > 0:
        print(f"\n  TOP CAMINOS CAUSALES A ANOMALY SCORE:")
        print(f"  {'─'*60}")

        adjacency = dag["adjacency"]
        variables = dag["variables"]
        var_idx = {v: i for i, v in enumerate(variables)}

        if "anomaly_score" in var_idx:
            target_idx = var_idx["anomaly_score"]
            n_vars = len(variables)

            all_paths_info = []
            for src_var in variables:
                if src_var == "anomaly_score":
                    continue
                src_idx = var_idx[src_var]
                paths = _find_all_directed_paths(adjacency, src_idx, target_idx, n_vars)

                for path in paths:
                    path_names = [variables[p] for p in path]
                    path_labels = [LABELS.get(v, v) for v in path_names]

                    # Calcular efecto del camino
                    path_effect = 1.0
                    for step in range(len(path) - 1):
                        # Buscar beta directo
                        found_beta = False
                        for _, erow in effects.iterrows():
                            if (erow["cause"] == variables[path[step]] and
                                    erow["effect"] == variables[path[step + 1]]):
                                path_effect *= erow["direct_beta"]
                                found_beta = True
                                break
                        if not found_beta:
                            path_effect = 0.0
                            break

                    all_paths_info.append({
                        "path": " -> ".join(path_labels),
                        "effect": path_effect,
                        "length": len(path),
                    })

            # Ordenar por efecto absoluto
            all_paths_info.sort(key=lambda x: abs(x["effect"]), reverse=True)

            for i, pi in enumerate(all_paths_info[:8]):
                direction = "+" if pi["effect"] > 0 else "-"
                print(f"    {i+1}. {pi['path']}")
                print(f"       Efecto: {pi['effect']:+.4f}")

    # ── Analisis contrafactual ──
    if len(counterfactuals) > 0:
        print(f"\n  ANALISIS CONTRAFACTUAL — Barrios anomalos:")
        print(f"  {'─'*60}")

        barrios = counterfactuals.groupby("barrio")

        for barrio_name, group in barrios:
            anomaly = group["anomaly_score"].iloc[0]
            print(f"\n    {barrio_name} (anomaly_score={anomaly:.3f}):")

            # Separar por categoria
            actionable = group[group["category"] == "ACCIONABLE"]
            structural = group[group["category"] == "ESTRUCTURAL"]

            total_actionable = actionable["fraction_explained"].sum()
            total_structural = structural["fraction_explained"].sum()

            for _, row in group.sort_values("fraction_explained", ascending=False).iterrows():
                pct = row["fraction_explained"] * 100
                if pct < 1.0:
                    continue
                cat_icon = "[A]" if row["category"] == "ACCIONABLE" else "[E]"
                print(f"      {cat_icon} {row['variable_label']:30s}: "
                      f"{pct:5.1f}% de la anomalia "
                      f"(delta={row['delta_anomaly_score']:+.3f})")

            if total_actionable > 0 or total_structural > 0:
                print(f"      Resumen: {total_actionable*100:.0f}% ACCIONABLE "
                      f"(reparar tuberias, contadores) | "
                      f"{total_structural*100:.0f}% ESTRUCTURAL (demografia, densidad)")

        # ── Separacion causas accionables vs estructurales ──
        print(f"\n  {'─'*60}")
        print(f"  CAUSAS ACCIONABLES vs ESTRUCTURALES (agregado):")
        print(f"  {'─'*60}")

        by_category = counterfactuals.groupby("category")["fraction_explained"].mean()
        for cat, frac in by_category.items():
            print(f"    {cat:15s}: {frac*100:.1f}% de la anomalia (media)")

        print(f"\n  ACCIONABLE = intervenciones posibles:")
        print(f"    - Reparar tuberias (reducir ANR/perdida fisica)")
        print(f"    - Reemplazar contadores (mejorar medicion)")
        print(f"    - Investigar desviaciones de tendencia")
        print(f"\n  ESTRUCTURAL = factores no modificables a corto plazo:")
        print(f"    - Demografia (envejecimiento del barrio)")
        print(f"    - Densidad poblacional")

    # ── Insight clave ──
    print(f"\n  {'='*60}")
    print(f"  INSIGHT CLAVE:")
    if len(effects) > 0 and len(counterfactuals) > 0:
        # Encontrar la causa accionable mas importante
        actionable_causes = counterfactuals[
            counterfactuals["category"] == "ACCIONABLE"
        ]
        if len(actionable_causes) > 0:
            top_actionable = (actionable_causes.groupby("variable_label")
                              ["fraction_explained"].mean()
                              .sort_values(ascending=False))
            if len(top_actionable) > 0:
                top_var = top_actionable.index[0]
                top_pct = top_actionable.iloc[0] * 100
                print(f"    La causa accionable principal es '{top_var}' "
                      f"({top_pct:.0f}% de la anomalia en promedio)")
                print(f"    → Priorizar intervenciones sobre este factor")

        structural_causes = counterfactuals[
            counterfactuals["category"] == "ESTRUCTURAL"
        ]
        if len(structural_causes) > 0:
            top_structural = (structural_causes.groupby("variable_label")
                              ["fraction_explained"].mean()
                              .sort_values(ascending=False))
            if len(top_structural) > 0:
                top_var_s = top_structural.index[0]
                top_pct_s = top_structural.iloc[0] * 100
                print(f"    Factor estructural dominante: '{top_var_s}' "
                      f"({top_pct_s:.0f}% promedio)")
                print(f"    → Este factor requiere politicas a largo plazo")
    else:
        print(f"    Datos insuficientes para conclusiones causales robustas")

    print(f"  {'='*60}\n")


# ─────────────────────────────────────────────────────────────────
# CLI: ejecutar directamente
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    csv_path = Path(__file__).parent / "results_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} no encontrado")
        sys.exit(1)

    print(f"Cargando datos de {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} filas, {len(df.columns)} columnas")

    results = run_causal_analysis(df)
    causal_summary(results)
