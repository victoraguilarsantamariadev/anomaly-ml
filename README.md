# AquaGuard AI — Deteccion de Anomalias Hidricas

Sistema de deteccion de anomalias en el consumo de agua para la ciudad de Alicante.

**Objetivo:** Detectar barrios con consumo anomalo (fugas, fraude, averias) analizando 4.3M lecturas de 2020-2025 (6 anos) con 6 modelos de ML/estadistica (de 14 probados, 8 descartados por ablation study) combinados en un ensemble inteligente.

---

## Como funciona (explicacion simple)

### El problema

AMAEM tiene datos de consumo de agua de 42 barrios de Alicante, mes a mes, durante 6 anos (2020-2025). Necesitan saber:
- Que barrios tienen consumo sospechoso (posible fraude o fugas)
- Con que confianza pueden afirmarlo
- Donde priorizar inspecciones para maximizar ahorro

### La solucion

AquaGuard AI usa **6 "detectives" diferentes** (modelos) que analizan el consumo desde perspectivas distintas. Se probaron 14, pero un ablation study descarto 8 que no aportaban o restaban fiabilidad. Cuando varios de los 6 supervivientes coinciden en que algo es raro, la confianza sube. Es como pedir segunda opinion medica: si varios doctores independientes dicen que hay un problema, probablemente lo hay.

### Los 6 modelos activos ("detectives")

| # | Nombre | Que hace | En simple | Peso |
|---|--------|----------|-----------|------|
| M2 | IsolationForest | Compara cada barrio con todos los demas | "Este barrio se comporta raro comparado con sus vecinos" | ~32% |
| M14 | VAE | Version probabilistica del autoencoder | El modelo MAS importante del sistema | ~28% |
| M13 | Autoencoder | Red neuronal que comprime y reconstruye | "No logro reconstruir este dato → es anomalo" | ~17% |
| M5b | IQR | Busca valores extremos con cuartiles | Robusto a datos raros | ~6% |
| M8 | ANR | Agua No Registrada | "Entra mas agua de la que se factura" | ~3% |
| M9 | NMF | Descompone patrones en componentes | "El patron de consumo no encaja" | ~3% |

**Modelos descartados (ablation study demostro que restaban fiabilidad):**
- ~~M5a 3-sigma~~, ~~M6 Chronos (Amazon)~~, ~~M7 Prophet (Facebook)~~ — peso=0 en el ensemble

### Como se combinan

1. **Cada modelo vota** si un barrio-mes es anomalo o no
2. **Weighted Voting:** Cada modelo vota con peso proporcional a su contribucion marginal (delta AUC-PR del ablation study). VAE + Autoencoder + M2 = ~77% del peso total
3. **Stacking:** Un GradientBoosting (GBM) aprende que combinaciones de modelos son mas fiables. Es el que realmente decide
4. **Ablation Study:** Mide que modelos aportan y cuales son ruido (leave-one-out, walk-forward). Los deltas se usan como pesos de voting
5. **Conformal Prediction:** Cada deteccion viene con un p-valor (0 a 1). Usa distancia Mahalanobis con Ledoit-Wolf shrinkage
6. **SHAP:** Cada deteccion viene con una explicacion ("alto consumo nocturno + infraestructura vieja")

### Resultado final

Cada barrio-mes recibe:
- **Alert color:** ROJO (muy sospechoso), NARANJA, AMARILLO, VERDE (normal)
- **Ensemble score:** 0 a 1 (probabilidad de anomalia)
- **P-valor conformal:** confianza estadistica
- **Explicacion SHAP:** por que se marco

---

## Como ejecutar

### 1. Instalar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
python run_all_models.py
```

Esto:
- Carga los datos de `data/datos-hackathon-amaem.xlsx-set-de-datos-.csv`
- Calcula features (consumo relativo, z-scores, tendencias, etc.)
- Ejecuta los 6 modelos activos (de 14 probados)
- Combina resultados con weighted voting + stacking + conformal
- Genera `results_full.csv` (648 filas x 75 columnas)
- Genera `ablation_results.csv` (importancia de cada modelo)

### 3. Generar el informe HTML

```bash
python generate_report.py
```

Genera `report.html` (133 KB) con 13 secciones: resumen, metodologia, resultados, dossiers de barrios, impacto economico, validacion contra fraude real, correlaciones, calibracion, robustez, ranking, out-of-sample 2025, y null permutation test.

### 4. Lanzar el dashboard interactivo

```bash
streamlit run dashboard.py
```

Dashboard con 7 pestanas: KPIs, Mapa de Alicante, Timeline por barrio, Validacion, AquaCare (impacto social), Los Modelos (14 probados, 6 activos), y Fiabilidad (pruebas visuales de validacion).

---

## Estructura del proyecto

### Archivos principales (los que importan)

```
anomaly-ml/
│
├── run_all_models.py          ← PIPELINE PRINCIPAL. Ejecuta todo.
├── generate_report.py         ← Genera el informe HTML
├── dashboard.py               ← Dashboard interactivo Streamlit
│
├── monthly_features.py        ← Calculo de features (z-scores, tendencias, etc.)
├── advanced_ensemble.py       ← Combina modelos: voting + stacking + conformal + SHAP
├── pseudo_ground_truth.py     ← Crea pseudo-etiquetas para evaluar (sin datos reales de fraude)
├── cross_validate_fraud.py    ← Valida contra datos reales de cambios de contador
│
├── results_full.csv           ← RESULTADO: 648 filas con todas las predicciones
├── ablation_results.csv       ← Importancia de cada modelo
├── report.html                ← Informe HTML generado
│
├── data/                      ← Datos de AMAEM (consumo, infraestructura, GIS)
├── requirements.txt           ← Dependencias Python
└── venv/                      ← Entorno virtual
```

### Archivos secundarios (detectores individuales)

```
├── chronos_detector.py        ← M6: Amazon Chronos
├── prophet_detector.py        ← M7: Facebook Prophet
├── autoencoder_detector.py    ← M13: Autoencoder
├── vae_detector.py            ← M14: VAE
├── nightflow_detector.py      ← M9: NMF
├── meter_readings_detector.py ← M10: Lecturas individuales
├── fraud_detector.py          ← M12: Meta-modelo de fraude
├── spatial_detector.py        ← Clasificacion espacial
├── changepoint_detector.py    ← Deteccion de cambios estructurales
```

### Archivos de soporte (no necesarios para ejecutar)

```
├── model_cv.py                ← Cross-validation temporal con tests estadisticos
├── tune_models.py             ← Tuning de hiperparametros
├── sensitivity_analysis.py    ← Analisis de sensibilidad
├── gis_features.py            ← Features GIS (infraestructura de red)
├── gis_utils.py               ← Utilidades para parsear GeoJSON
├── hydraulic_twin.py          ← Gemelo digital hidraulico
├── causal_analysis.py         ← Inferencia causal
├── transfer_entropy.py        ← Entropia de transferencia
├── tda_detector.py            ← Topological Data Analysis
├── wasserstein_detector.py    ← Distancia Wasserstein
├── welfare_detector.py        ← Deteccion de vulnerabilidad social
├── counterfactual_explainer.py ← Explicaciones contrafactuales
├── mlops_monitor.py           ← Monitorizacion de drift
```

---

## Datos

Los datos estan en `data/` y vienen de AMAEM:

| Archivo | Que contiene |
|---------|-------------|
| `datos-hackathon-amaem.xlsx-set-de-datos-.csv` | Consumo mensual por barrio (2022-2024) |
| `cambios-de-contador-solo-alicante_*.csv` | Cambios de contador (incluye fraude real) |
| `contadores-telelectura-*.csv` | Contadores con telelectura instalados |
| `_consumos_alicante_regenerada_*.csv` | Consumo de agua regenerada |
| `padron_elderly_barrios_2025.csv` | Poblacion mayor por barrio |
| `sectores_de_consumo.json` | 183 sectores hidraulicos (GIS) |
| `entidades_de_poblacion.json` | Limites de barrios (GIS) |
| `bocasriego_hidrantes.json` | Hidrantes y bocas de riego |
| `tuberias*.json` | Red de tuberias |

---

## Resultados clave

### Metricas del sistema

| Metrica | Valor |
|---------|-------|
| AUC-PR (full ensemble) | 0.893 |
| Precision (stacking >= 0.5) | 0.543 |
| Recall | 0.643 |
| F1 | 0.589 |
| ECE (calibracion) | 0.037 (EXCELENTE) |
| Lift vs random | 4.6x |
| Barrios analizados | 27 |
| Modelos activos | 6 (de 14 probados) |

### Distribucion de alertas

- VERDE (normal): ~67%
- AMARILLO (atencion): ~21%
- NARANJA (sospechoso): ~7%
- ROJO (critico): ~5%

### Modelos mas importantes (ablation study, cargado dinamicamente de ablation_results.csv)

| Modelo | Delta AUC-PR | Veredicto |
|--------|-------------|-----------|
| M2 IsoForest | +0.086 | ESENCIAL |
| M14 VAE | +0.078 | ESENCIAL |
| M13 Autoencoder | +0.049 | ESENCIAL |
| M5b IQR | +0.024 | UTIL |
| Resto (5 modelos) | <=0.003 | REDUNDANTE |

Nota: Los pesos se cargan automaticamente desde `ablation_results.csv` en cada ejecucion. Los numeros de arriba corresponden a la ultima ejecucion.

### Impacto economico estimado

- Ahorro potencial: ~EUR 518,000/ano
- ROI de inspecciones: 136x
- Top 30% barrios captura 80% de anomalias

---

## Validacion (como sabemos que funciona)

El sistema incluye 22 capas de validacion (todos los p-valores externos con permutation test, 10K shuffles):

**Validacion interna (contra pseudo-labels):**
1. **Bootstrap CI:** Intervalos de confianza al 95% sobre todas las metricas (1000 iteraciones)
2. **Baseline comparison:** El ensemble supera al mejor modelo individual, a seasonal naive, y al azar
3. **Tests estadisticos:** Friedman (p<0.05: modelos son significativamente diferentes) + Wilcoxon pairwise
4. **Ablation study:** Contribucion marginal de cada modelo medida con leave-one-out
5. **Reliability diagram:** Curva de calibracion mostrando que las probabilidades son honestas

**Validacion externa (contra datos reales independientes):**
6. **Correlacion con fraude real:** M2 tiene r=+0.540 con datos reales de cambios de contador por fraude
7. **Caudal nocturno (MNF):** rho=+0.297, hit-rate 50%. Los barrios con mas exceso de caudal entre 2-4 AM (= fugas reales) coinciden en un 50% con nuestros top anomalos
8. **Riesgo de infraestructura:** rho=-0.08 (NO significativa). El sistema NO detecta simplemente "infraestructura vieja" — detecta anomalias de consumo reales, no proxies triviales
9. **Cobertura smart meters:** rho=+0.01 (NO significativa). Las detecciones no dependen de si el barrio tiene contadores inteligentes o no
10. **Balance hidrico:** rho=+0.413 (p_perm=0.056, MARGINAL), hit-rate 70%. Compara agua que ENTRA al sector (caudal medido fisicamente) vs agua FACTURADA. 7 de 10 barrios con mas perdidas hidricas coinciden con nuestros top anomalos
11. **Agua regenerada (control negativo):** rho=+0.43, resultado inesperado — correlaciona porque barrios con mas riego publico (Playa de San Juan) tambien tienen mas variacion estacional
12. **Lecturas individuales de contadores:** rho=+0.794, p=0.003 (SIGNIFICATIVA). 4.3M lecturas de m3-registrados (6 anos: 2020-2025). Los meses con mas contadores sospechosos (zeros, negativos, dias anormales) coinciden con meses de mas anomalias detectadas
13. **Weather deconfounding:** Partial rho=+0.51 tras controlar por temperatura, precipitacion y turismo (AEMET). Las anomalias PERSISTEN tras eliminar efectos climaticos → NO son artefactos del clima. rho(ensemble,temp)=-0.66 (mas anomalias en INVIERNO, no verano)
14. **Fisher's combined test:** Combina p-valores de MNF + Balance hidrico + Lecturas contadores. Fisher's p=0.002 — ALTAMENTE SIGNIFICATIVO
15. **Benjamini-Hochberg FDR:** Correccion por test multiple sobre las 7 validaciones. Lecturas contadores (q=0.016) SOBREVIVE la correccion — resultado robusto
16. **Poda de modelos dañinos:** Ablation study demuestra que Prophet, 3-sigma y Chronos RESTAN AUC-PR. Eliminados del ensemble (peso=0). Solo quedan 6 modelos que aportan
17. **Stable Core (4 barrios):** Barrios detectados por TODOS los metodos simultaneamente (top-25% ensemble + top-25% stacking + conformal p<0.05 + >=3 modelos): Virgen del Carmen, Dispersos, Colonia Requena, Playa de San Juan
18. **Null Permutation Test:** p=0.001, Z=18.9 — los barrios detectados son COMPLETAMENTE separables del azar. 1000 permutaciones de etiquetas de barrio nunca producen scores comparables
19. **Bootstrap Stable Core (500 resamples):** 3 barrios ultra-estables (>80%): Dispersos (100%), Colonia Requena (95%), Virgen del Carmen (91%)
20. **Moran's I (autocorrelacion espacial):** I=-0.024, p=0.50 — distribucion espacial aleatoria (anomalias no concentradas en zona especifica, sugiere causas distribuidas como fraude/errores)
21. **Out-of-sample temporal 2025:** rho=+0.632, p=0.027 (SIGNIFICATIVA). Anomalias mensuales de 2022-2024 predicen patrones de 2025 (767K lecturas no vistas). Las anomalias son ESTRUCTURALES, no artefactos del periodo de entrenamiento
22. **Micro-validacion codigos postales:** rho=-0.236, p=0.377 (NO significativa). Correlacion entre CV de consumo por codigo postal y ensemble_score por barrio. Mapping postal→barrio demasiado grueso para ser concluyente

**AquaCare — Deteccion de fugas silenciosas en hogares de mayores:**
- Cruza 192K contadores residenciales (con barrio) con padron elderly de Alicante 2025
- Top barrios en riesgo: Playa de San Juan (~57 contadores), Albufereta (~76), Ensanche Diputacion (~54)
- Score de riesgo = f(contadores viejos, % mayores solos, tamaño barrio)
- **Score dual:** vulnerability_score (demografico) + consumption_risk_score (dinamico) → combined silent_leak_risk
- **5 validaciones AquaCare:** V4 permutation p=0.001 (targeting real). V1-V3 ahora positivas (antes negativas) gracias a features dinamicos, pero no significativas aun
- Playa San Juan ultra-robusto (86% en 200 configs Dirichlet)

**Interpretacion honesta:** Las detecciones son estadisticamente separables del azar (null test p=0.001). Correlacionan con 4.3M lecturas individuales de 6 anos (rho=0.79, q_BH=0.016), fugas nocturnas (hit-rate 50%), y perdidas hidricas (hit-rate 70%). No son artefactos del clima. Fisher's p=0.002. 3 barrios son ultra-estables en bootstrap (100%, 95%, 91%). **Out-of-sample 2025: rho=+0.63, p=0.027** — las anomalias son ESTRUCTURALES, confirmadas en datos no vistos. La unica limitacion: no distinguimos fraude de fugas de errores de medicion sin inspeccion de campo

---

## Decisiones tecnicas clave

### Por que weighted voting con pesos del ablation + stacking GBM?
Cada modelo vota con peso = su delta AUC-PR del ablation study (contribucion marginal medida). Esto es data-driven: modelos que aportan mas pesan mas. La oscilacion entre corridas se evita porque el ablation study no usa `ensemble_score` como feature (rompiendo la dependencia circular). El GBM meta-learner (stacking) hace la ponderacion final no-lineal.

### Por que pseudo-labels en vez de labels reales?
No hay datos etiquetados de "este barrio tiene fraude". Construimos pseudo-labels combinando 3 senales independientes de los modelos: antigueedad de infraestructura, desviacion del grupo, y tasa de reemplazo de contadores.

### Por que walk-forward y no cross-validation normal?
Los datos son temporales. Usar datos de 2024 para predecir 2023 seria hacer trampa. Walk-forward entrena solo con pasado y predice futuro, como en produccion real.

### Por que conformal prediction?
Permite dar un p-valor a cada deteccion. Un p-valor < 0.05 significa "con 95% de confianza, esto NO es consumo normal". Es el mismo estandar que se usa en ciencia y medicina. Usa distancia Mahalanobis con estimador de covarianza Ledoit-Wolf (shrinkage optimo).

### Nada esta hardcodeado
- **Ablation weights:** Se cargan de `ablation_results.csv` en cada ejecucion
- **Parametros tuneados:** Se cargan de `tuned_params_cv.json` (contamination, IQR multiplier) si existe
- **Thresholds:** Optimizados por F1 en walk-forward validation, no fijos
- **Arquitectura neural:** VAE (64→32→latent=16, beta=2.0) y Autoencoder (32→16→8→16→32)

### Mejoras quant-level aplicadas (3 rondas)

**Ronda 1:** Anti-leak (look-ahead bias fix, isotonic calibration fix, pseudo-label circularity break)
**Ronda 2:** Arquitectura (VAE ampliado + beta-VAE, autoencoder ampliado, persistence features, Mahalanobis conformal)
**Ronda 3:** Anti-hardcoding (pesos dinamicos desde CSV, auto-carga tuned params, Ledoit-Wolf shrinkage, voting con pesos del ablation anti-oscilacion)
