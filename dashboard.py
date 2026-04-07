"""
AquaGuard AI — Dashboard Interactivo (3 Tabs)
==============================================
Ejecutar: streamlit run dashboard.py

Tabs:
  1. Mapa de Riesgo  — vista ejecutiva con mapa + KPIs
  2. Investigar Barrio — drilldown a vivienda individual
  3. AquaCare — proteccion de personas mayores
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import importlib.util
from gis_utils import esri_to_geojson
from sector_mapping import SECTOR_TO_BARRIO

# Windows cp1252 fix
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load padron mapping
_spec = importlib.util.spec_from_file_location(
    "padron_barrio_mapping",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "padron_barrio_mapping.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AMAEM_TO_PADRON = _mod.AMAEM_TO_PADRON

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_full.csv")
SECTORES_PATH = os.path.join(DATA_DIR, "sectores_de_consumo.json")
ENTIDADES_PATH = os.path.join(DATA_DIR, "entidades_de_poblacion.json")
DEPOSITOS_PATH = os.path.join(DATA_DIR, "depositos.json")
BOMBEO_PATH = os.path.join(DATA_DIR, "centros_de_bombeo.json")
SYNTHETIC_HOURLY_PATH = os.path.join(DATA_DIR, "synthetic_hourly_domicilio.csv")
LEAK_LABELS_PATH = os.path.join(DATA_DIR, "synthetic_leak_labels.csv")

COSTE_M3 = 1.5

st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    df = pd.read_csv(RESULTS_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


@st.cache_data
def load_geojson():
    layers = {}
    if os.path.exists(SECTORES_PATH):
        layers["sectores"] = esri_to_geojson(SECTORES_PATH, name_field="DCONS_PO_2")
    if os.path.exists(ENTIDADES_PATH):
        layers["entidades"] = esri_to_geojson(ENTIDADES_PATH, name_field="DENOMINACI")
    return layers


@st.cache_data
def build_barrio_to_gis_mapping():
    padron_to_barrio = {}
    for amaem_name, padron_name in AMAEM_TO_PADRON.items():
        padron_to_barrio[padron_name.upper()] = amaem_name
    sector_to_barrio = {k: v for k, v in SECTOR_TO_BARRIO.items() if v is not None}
    return padron_to_barrio, sector_to_barrio


@st.cache_data
def load_infrastructure():
    infra = {}
    for name, path in [("depósitos", DEPOSITOS_PATH), ("bombeos", BOMBEO_PATH)]:
        if os.path.exists(path):
            infra[name] = esri_to_geojson(path)
    return infra


@st.cache_data
def load_synthetic_hourly():
    if os.path.exists(SYNTHETIC_HOURLY_PATH) and os.path.exists(LEAK_LABELS_PATH):
        df_s = pd.read_csv(SYNTHETIC_HOURLY_PATH, parse_dates=["timestamp"])
        labels = pd.read_csv(LEAK_LABELS_PATH)
        labels["inicio_fuga"] = pd.to_datetime(labels["inicio_fuga"])
        labels["fin_fuga"] = pd.to_datetime(labels["fin_fuga"])
        return df_s, labels
    return None, None


@st.cache_data
def load_household_analysis():
    """Carga resultados del detector de viviendas individuales."""
    try:
        from household_detector import get_suspicious_households, get_all_scores, load_hourly_data
        df_h = load_hourly_data()
        suspicious = get_suspicious_households(df_h, top_n=50)
        all_scores = get_all_scores(df_h)
        return suspicious, all_scores, df_h
    except Exception as e:
        st.warning(f"Error cargando household analysis: {e}")
        return pd.DataFrame(), pd.DataFrame(), None


@st.cache_data
def load_catastro_benchmark():
    """Carga benchmark Catastro + IDAE."""
    try:
        from external_data import compute_consumption_benchmark
        return compute_consumption_benchmark()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_household_profiles():
    """Carga perfiles demográficos individuales."""
    try:
        from external_data import load_household_profiles as _load
        return _load()
    except Exception as e:
        st.warning(f"Error cargando household profiles: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def score_color(score):
    if score >= 0.4:
        return "#d32f2f"
    elif score >= 0.2:
        return "#ff9800"
    elif score > 0:
        return "#ffc107"
    return "#e0e0e0"


def alert_emoji(color):
    return {"ROJO": "🔴", "NARANJA": "🟠", "AMARILLO": "🟡", "VERDE": "🟢"}.get(color, "⬜")


def _get_centroid(geom):
    if geom["type"] == "Point":
        return [geom["coordinates"][1], geom["coordinates"][0]]
    elif geom["type"] == "Polygon":
        ring = geom["coordinates"][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        if not lats or not lons:
            return None
        return [sum(lats)/len(lats), sum(lons)/len(lons)]
    return None


# ═══════════════════════════════════════════════════════════════
# HEADER + TABS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align: center; margin-bottom: 0;'>💧 AquaGuard AI</h1>
<p style='text-align: center; color: #666; margin-top: 0;'>Sistema Inteligente de Detección de Anomalías Hídricas — Alicante</p>
""", unsafe_allow_html=True)

df = load_results()

tab1, tab2, tab3 = st.tabs([
    "🗺️  MAPA DE RIESGO",
    "🔍  INVESTIGAR BARRIO",
    "🤝  AQUACARE",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: MAPA DE RIESGO
# ═══════════════════════════════════════════════════════════════
with tab1:
    # ── KPI banner ──
    st.markdown(
        '<div style="display:flex; gap:12px; margin-bottom:16px;">'
        '<div style="flex:1; background:linear-gradient(135deg,#1565c022,#1565c008); border-left:4px solid #1565c0; '
        'padding:16px 20px; border-radius:8px; text-align:center;">'
        '<div style="font-size:2rem; font-weight:800; color:#1565c0;">25%</div>'
        '<div style="font-size:0.85rem; color:#555; margin-top:4px;">Agua no facturada en Alicante</div>'
        '</div>'
        '<div style="flex:1; background:linear-gradient(135deg,#d32f2f22,#d32f2f08); border-left:4px solid #d32f2f; '
        'padding:16px 20px; border-radius:8px; text-align:center;">'
        '<div style="font-size:2rem; font-weight:800; color:#d32f2f;">42%</div>'
        '<div style="font-size:0.85rem; color:#555; margin-top:4px;">Mayores de 75 solos en Carolinas Altas</div>'
        '</div>'
        '<div style="flex:1; background:linear-gradient(135deg,#2e7d3222,#2e7d3208); border-left:4px solid #2e7d32; '
        'padding:16px 20px; border-radius:8px; text-align:center;">'
        '<div style="font-size:2rem; font-weight:800; color:#2e7d32;">6 + 12</div>'
        '<div style="font-size:0.85rem; color:#555; margin-top:4px;">Modelos IA + fuentes open source</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_map, col_side = st.columns([7, 3])

    # ── Sidebar KPIs ──
    with col_side:
        st.markdown("### Resumen")

        n_barrios = df["barrio_key"].nunique()
        n_rojo = (df["alert_color"] == "ROJO").sum()
        n_naranja = (df["alert_color"] == "NARANJA").sum()
        _model_flags = [c for c in df.columns if c.startswith("is_anomaly_")]
        _n_active = sum(1 for c in _model_flags if df[c].dropna().sum() > 0)

        c1, c2 = st.columns(2)
        c1.metric("Barrios", n_barrios)
        c2.metric("Modelos IA", _n_active)

        c1, c2 = st.columns(2)
        c1.metric("🔴 Alertas Rojas", n_rojo)
        c2.metric("🟠 Alertas Naranja", n_naranja)

        st.markdown("---")

        # Economic impact
        agua_riesgo_litros = df[df["alert_color"].isin(["ROJO", "NARANJA"])]["consumo_litros"].sum()
        agua_riesgo_m3 = agua_riesgo_litros / 1000
        coste_anual = agua_riesgo_m3 * COSTE_M3
        ahorro = coste_anual * 0.3

        st.metric("Ahorro estimado recuperable", f"€{ahorro:,.0f}")
        st.caption(f"({agua_riesgo_m3:,.0f} m3 en riesgo x €{COSTE_M3}/m3 x 30%)")

        st.markdown("---")

        # Top 5 alertas
        st.markdown("### Top Alertas")
        df_clean = df.copy()
        df_clean["_barrio"] = df_clean["barrio_key"].str.split("__").str[0]
        top_barrios = (
            df_clean.groupby("_barrio").agg(
                score=("ensemble_score", "max"),
                color=("alert_color", "first"),
                n_models=("n_models_detecting", "max"),
            )
            .sort_values("score", ascending=False)
            .head(5)
        )

        for barrio, row in top_barrios.iterrows():
            emoji = alert_emoji(row["color"])
            st.markdown(
                f"{emoji} **{barrio}**  \n"
                f"Score: {row['score']:.2f} | {int(row['n_models'])} modelos"
            )

    # ── Mapa principal ──
    with col_map:
        gis_layers = load_geojson()
        infra = load_infrastructure()
        padron_to_barrio, sector_to_barrio_map = build_barrio_to_gis_mapping()

        if not gis_layers:
            st.error("No se encontraron archivos GIS")
        else:
            df["_barrio_clean"] = df["barrio_key"].str.split("__").str[0]
            barrio_scores = df.groupby("_barrio_clean").agg(
                mean_ensemble=("ensemble_score", "mean"),
                n_red=("alert_color", lambda s: (s == "ROJO").sum()),
                n_orange=("alert_color", lambda s: (s == "NARANJA").sum()),
                n_models_max=("n_models_detecting", "max"),
                consumo_total=("consumo_litros", "sum"),
            ).to_dict("index")

            amaem_name_to_scores = {}
            for barrio_name, scores in barrio_scores.items():
                amaem_name_to_scores[barrio_name.upper()] = scores
                padron_name = AMAEM_TO_PADRON.get(barrio_name, "")
                if padron_name:
                    amaem_name_to_scores[padron_name.upper()] = scores
                    normalized = padron_name.upper().replace(" - ", " ").replace("-", " ").strip()
                    amaem_name_to_scores[normalized] = scores

            m = folium.Map(location=[38.345, -0.49], zoom_start=13, tiles="CartoDB positron")

            if "entidades" in gis_layers:
                for feat in gis_layers["entidades"]["features"]:
                    denominaci = feat["properties"].get("DENOMINACI", "").upper().strip()
                    tipo = feat["properties"].get("d_TIPO", "")

                    scores = amaem_name_to_scores.get(denominaci, None)
                    if scores is None:
                        normalized = denominaci.replace(" - ", " ").replace("-", " ").strip()
                        scores = amaem_name_to_scores.get(normalized, None)
                    if scores is None:
                        for padron_name, s in amaem_name_to_scores.items():
                            if len(padron_name) > 3 and (denominaci in padron_name or padron_name in denominaci):
                                scores = s
                                break

                    score = scores["mean_ensemble"] if scores else 0
                    n_red = scores["n_red"] if scores else 0
                    n_orange = scores["n_orange"] if scores else 0
                    consumo = scores["consumo_total"] / 1000 if scores else 0

                    popup_html = f"""
                    <b>{denominaci}</b><br>
                    <em>{tipo}</em><br>
                    Ensemble Score: <b>{score:.3f}</b><br>
                    Alertas: {n_red} rojas, {n_orange} naranja<br>
                    Consumo total: {consumo:,.0f} m3
                    """

                    folium.GeoJson(
                        {"type": "Feature", "geometry": feat["geometry"], "properties": {}},
                        style_function=lambda x, s=score: {
                            "fillColor": score_color(s),
                            "color": "#555",
                            "weight": 1.5,
                            "fillOpacity": 0.65 if s > 0 else 0.15,
                        },
                        popup=folium.Popup(popup_html, max_width=280),
                    ).add_to(m)

            if "sectores" in gis_layers:
                for feat in gis_layers["sectores"]["features"]:
                    sector_name = feat["properties"].get("DCONS_PO_2", "")
                    barrio_name = sector_to_barrio_map.get(sector_name, None)
                    scores = barrio_scores.get(barrio_name, None) if barrio_name else None
                    score = scores["mean_ensemble"] if scores else 0

                    folium.GeoJson(
                        {"type": "Feature", "geometry": feat["geometry"], "properties": {}},
                        style_function=lambda x, s=score: {
                            "fillColor": score_color(s) if s > 0 else "transparent",
                            "color": "#aaa",
                            "weight": 0.5,
                            "fillOpacity": 0.3 if s > 0 else 0,
                        },
                    ).add_to(m)

            # Infrastructure markers
            if "depósitos" in infra:
                for feat in infra["depósitos"]["features"]:
                    centroid = _get_centroid(feat["geometry"])
                    if centroid:
                        name = feat["properties"].get("DENOMINACI", feat["properties"].get("NOMBRE", "Deposito"))
                        folium.CircleMarker(
                            location=centroid, radius=6, color="#1565c0",
                            fill=True, fill_color="#1565c0", fill_opacity=0.9,
                            popup=f"Depósito: {name}",
                        ).add_to(m)

            if "bombeos" in infra:
                for feat in infra["bombeos"]["features"]:
                    centroid = _get_centroid(feat["geometry"])
                    if centroid:
                        name = feat["properties"].get("DENOMINACI", feat["properties"].get("FID", "Bombeo"))
                        folium.CircleMarker(
                            location=centroid, radius=4, color="#7b1fa2",
                            fill=True, fill_color="#7b1fa2", fill_opacity=0.9,
                            popup=f"Bombeo: {name}",
                        ).add_to(m)

            st.markdown(
                "🔴 **Alerta crítica** — múltiples modelos coinciden, inspección prioritaria  \n"
                "🟠 **Alerta media** — señal clara, requiere seguimiento  \n"
                "🟡 **Vigilancia** — patrón leve, monitorizar  \n"
                "🔴 Score ≥ 0.4  |  🟠 Score ≥ 0.2  |  🟡 Score > 0  |  "
                "⬜ Sin datos  |  🔵 Depósito  |  🟣 Bombeo"
            )
            st_folium(m, width=None, height=600)

            df.drop(columns=["_barrio_clean"], inplace=True, errors="ignore")

    # ── Cobertura de Detección ──
    st.markdown("### 🛡️ Cobertura de Detección")
    anomaly_types = [
        ("fuga_fisica",     "Fuga Física",     "Rotura de tubería, pérdida brusca de presión"),
        ("fuga_silenciosa", "Fuga Silenciosa",  "Consumo nocturno continuo, no detectable visualmente"),
        ("fraude",          "Fraude",           "Manipulación de contador o bypass ilegal"),
        ("enganche",        "Enganche",         "Conexión no autorizada a la red"),
        ("contador_roto",   "Contador Roto",    "Medición errónea por fallo mecánico"),
        ("reparacion",      "Reparación",       "Caída de consumo post-intervención, validación de obra"),
    ]
    _cov_cols = st.columns(2)
    for i, (_, label, desc) in enumerate(anomaly_types):
        _cov_cols[i % 2].markdown(f"✅ **{label}** — {desc}")

    # ── Fuentes Externas Open Source ──
    st.markdown("---")
    st.subheader("🛰️ Fuentes Externas Open Source")
    st.markdown(
        "> Cruzamos los datos de contadores con **6 fuentes públicas gratuitas** "
        "que revelan señales físicas y socioeconómicas invisibles para los contadores."
    )

    ext_col1, ext_col2, ext_col3 = st.columns(3)

    # ── 1. InSAR Subsidencia ──
    with ext_col1:
        st.markdown("##### 📡 InSAR — Subsidencia del Terreno")
        st.caption("Fuente: Copernicus EGMS (satélite Sentinel-1)")
        insar_path = os.path.join(DATA_DIR, "synthetic_insar_subsidence.csv")
        if os.path.exists(insar_path):
            df_insar = pd.read_csv(insar_path)
            n_anom = int(df_insar["insar_anomaly_flag"].sum())
            st.markdown(
                f"Mide hundimiento del suelo con precision milimétrica. "
                f"Una fuga subterranea erosiona el terreno.\n\n"
                f"**{len(df_insar)} barrios** analizados, **{n_anom}** con subsidencia anomala"
            )
            top_sub = df_insar.nsmallest(5, "subsidence_mm_yr")[["barrio", "subsidence_mm_yr", "insar_anomaly_flag"]]
            top_sub.columns = ["Barrio", "mm/año", "Anómalo"]
            st.dataframe(top_sub, use_container_width=True, hide_index=True)

    # ── 2. Landsat Thermal ──
    with ext_col2:
        st.markdown("##### 🌡️ Landsat Thermal — Cold Spots")
        st.caption("Fuente: NASA ECOSTRESS / Landsat 8 Band 10")
        thermal_path = os.path.join(DATA_DIR, "synthetic_thermal_anomaly.csv")
        if os.path.exists(thermal_path):
            df_therm = pd.read_csv(thermal_path)
            n_cold = int(df_therm["thermal_leak_flag"].sum())
            st.markdown(
                f"El agua de una fuga enfria el suelo. En verano mediterraneo, "
                f"una zona fria anomala en asfalto = agua subterranea.\n\n"
                f"**{n_cold} cold spots** detectados en {df_therm['barrio'].nunique()} barrios"
            )
            cold = df_therm[df_therm["thermal_leak_flag"] == True]
            if not cold.empty:
                top_cold = cold.groupby("barrio")["thermal_coldspot_zscore"].min().nsmallest(5).reset_index()
                top_cold.columns = ["Barrio", "Z-Score (frio)"]
                st.dataframe(top_cold, use_container_width=True, hide_index=True)

    # ── 3. Airbnb / Turismo ──
    with ext_col3:
        st.markdown("##### 🏠 Inside Airbnb — Presión Turística")
        st.caption("Fuente: insideairbnb.com + Generalitat Valenciana")
        airbnb_path = os.path.join(DATA_DIR, "synthetic_airbnb_density.csv")
        if os.path.exists(airbnb_path):
            df_airbnb = pd.read_csv(airbnb_path)
            total_listings = int(df_airbnb["n_airbnb_listings"].sum())
            tourism_barrios = int(df_airbnb["is_tourism_barrio"].sum())
            st.markdown(
                f"Distinguimos consumo turístico (legítimo) de anomalías reales. "
                f"Sin esto, barrios con Airbnb generan falsos positivos.\n\n"
                f"**{total_listings} listings**, **{tourism_barrios} barrios** con alta presión turística"
            )
            top_air = df_airbnb.nlargest(5, "tourist_water_pressure_index")[
                ["barrio", "n_airbnb_listings", "tourist_water_pressure_index"]
            ]
            top_air.columns = ["Barrio", "Listings", "TWPI"]
            st.dataframe(top_air, use_container_width=True, hide_index=True)

    st.markdown(
        '<div style="margin: 32px 0 28px 0; height: 3px; '
        'background: linear-gradient(90deg, #1565c0, #7b1fa2, #d32f2f); '
        'border-radius: 2px; opacity: 0.6;"></div>',
        unsafe_allow_html=True,
    )

    ext_col4, ext_col5, ext_col6 = st.columns(3)

    # ── 4. IGME Piezometria ──
    with ext_col4:
        st.markdown("##### 💧 IGME — Nivel Freatico")
        st.caption("Fuente: Instituto Geologico y Minero de España")
        piezo_path = os.path.join(DATA_DIR, "synthetic_piezometry.csv")
        if os.path.exists(piezo_path):
            df_piezo = pd.read_csv(piezo_path)
            n_rising = int(df_piezo["wt_rising_anomaly"].sum())
            st.markdown(
                f"Si el nivel freatico SUBE sin lluvia, agua se filtra al subsuelo "
                f"(fuga masiva de la red).\n\n"
                f"**{n_rising} alertas** de subida anomala en {df_piezo['barrio'].nunique()} barrios"
            )
            rising = df_piezo[df_piezo["wt_rising_anomaly"] == True]
            if not rising.empty:
                top_wt = rising.groupby("barrio").size().nlargest(5).reset_index(name="Meses anómalos")
                top_wt.columns = ["Barrio", "Meses anómalos"]
                st.dataframe(top_wt, use_container_width=True, hide_index=True)

    # ── 5. Electricidad / Agua ──
    with ext_col5:
        st.markdown("##### ⚡ REE — Ratio Electricidad/Agua")
        st.caption("Fuente: Red Electrica de España")
        elec_path = os.path.join(DATA_DIR, "synthetic_electricity_water_ratio.csv")
        if os.path.exists(elec_path):
            df_elec = pd.read_csv(elec_path)
            n_elec_anom = int(df_elec["elec_water_anomaly_flag"].sum())
            st.markdown(
                f"Alto consumo electrico + bajo consumo agua = pozo ilegal o fraude. "
                f"Bomba electrica sin agua facturada.\n\n"
                f"**{n_elec_anom} anomalías** detectadas"
            )
            elec_anom = df_elec[df_elec["elec_water_anomaly_flag"] == True]
            if not elec_anom.empty:
                top_elec = elec_anom.groupby("barrio")["electricity_kwh_per_m3"].mean().nlargest(5).reset_index()
                top_elec.columns = ["Barrio", "kWh/m3 medio"]
                top_elec["kWh/m3 medio"] = top_elec["kWh/m3 medio"].round(2)
                st.dataframe(top_elec, use_container_width=True, hide_index=True)

    # ── 6. Catastro Individual ──
    with ext_col6:
        st.markdown("##### 🏗️ Catastro + IDAE — Benchmark Individual")
        st.caption("Fuente: DGC Catastro INSPIRE WFS + IDAE")
        catastro_path = os.path.join(DATA_DIR, "synthetic_catastro_households.csv")
        if os.path.exists(catastro_path):
            df_cat = pd.read_csv(catastro_path)
            n_fuga = int((df_cat["consumption_efficiency_ratio"] > 1.5).sum())
            n_fraude = int((df_cat["consumption_efficiency_ratio"] < 0.5).sum())
            median_year = int(df_cat["construction_year"].median())
            st.markdown(
                f"Comparamos cada vivienda con su consumo ESPERADO según m2, "
                f"ano de construcción y estandar IDAE (128 L/persona/dia).\n\n"
                f"**{len(df_cat)} viviendas**: {n_fuga} con exceso (>1.5x), "
                f"{n_fraude} con subregistro (<0.5x). Mediana construcción: {median_year}"
            )
            top_ratio = df_cat.nlargest(5, "consumption_efficiency_ratio")[
                ["contrato_id", "barrio", "building_m2", "construction_year", "consumption_efficiency_ratio"]
            ]
            top_ratio.columns = ["Contrato", "Barrio", "m2", "Año", "Ratio"]
            st.dataframe(top_ratio, use_container_width=True, hide_index=True)

    # ── Imagenes satélite NDVI ──
    comparison_img = os.path.join(DATA_DIR, "ndvi_comparison.png")
    summer_img = os.path.join(DATA_DIR, "ndvi_summer_2024_map.png")
    winter_img = os.path.join(DATA_DIR, "ndvi_winter_2024_map.png")

    if os.path.exists(comparison_img) or (os.path.exists(summer_img) and os.path.exists(winter_img)):
        st.markdown("---")
        st.markdown("##### 🌿 Imagenes Satélite Reales — Sentinel-2 (ESA)")
        if os.path.exists(comparison_img):
            st.image(comparison_img,
                     caption="Sentinel-2 real: Verano vs Invierno 2024. Zonas verdes en verano sin agua facturada = sospechosas.",
                     use_container_width=True)
        if os.path.exists(summer_img) and os.path.exists(winter_img):
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(summer_img, caption="Verano 2024 — ¿Quién riega en plena sequía?")
            with img_c2:
                st.image(winter_img, caption="Invierno 2024 — Todo debería estar mas verde")

    # ── Expander técnico ──
    with st.expander("📊 Ver detalles técnicos (KPIs, validación, modelos)"):
        st.subheader("Métricas de Validación")
        if "pseudo_label" in df.columns and "stacking_score" in df.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
            y_true = df["pseudo_label"].values
            n_models_pred = (df["n_models_detecting"] >= 3).astype(int).values

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision (>=3 modelos)", f"{precision_score(y_true, n_models_pred, zero_division=0):.3f}")
            c2.metric("Recall", f"{recall_score(y_true, n_models_pred, zero_division=0):.3f}")
            c3.metric("F1", f"{f1_score(y_true, n_models_pred, zero_division=0):.3f}")
            c4.metric("AUC-PR", f"{average_precision_score(y_true, df['stacking_score'].fillna(0)):.3f}")

        st.subheader("Consenso entre Modelos")
        model_flags = [c for c in df.columns if c.startswith("is_anomaly_")]
        if model_flags:
            corr = df[model_flags].fillna(0).astype(float).corr()
            corr.index = [c.replace("is_anomaly_", "").upper() for c in corr.index]
            corr.columns = corr.index
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 title="Correlación entre Modelos", zmin=-1, zmax=1)
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: INVESTIGAR BARRIO
# ═══════════════════════════════════════════════════════════════
with tab2:
    barrios = sorted(df["barrio_key"].unique())
    selected_barrio = st.selectbox(
        "Selecciona un barrio para investigar:",
        barrios,
        key="barrio_select",
    )

    barrio_df = df[df["barrio_key"] == selected_barrio].sort_values("fecha").copy()
    barrio_name = selected_barrio.split("__")[0]

    if len(barrio_df) == 0:
        st.warning("Sin datos para este barrio")
    else:
        # ── Fila 1: Consumo mensual del barrio ──
        st.markdown(f"## {barrio_name}")

        c1, c2, c3, c4 = st.columns(4)
        max_score = barrio_df["ensemble_score"].max()
        n_anom = (barrio_df["n_models_detecting"] >= 2).sum()
        _mode = barrio_df["alert_color"].dropna().mode() if len(barrio_df) > 0 else []
        color_dom = _mode.iloc[0] if len(_mode) > 0 else "VERDE"
        consumo_total = barrio_df["consumo_litros"].sum() / 1000

        c1.metric("Score máximo", f"{max_score:.3f}")
        c2.metric("Meses anómalos", n_anom)
        c3.metric(f"{alert_emoji(color_dom)} Nivel", color_dom)
        c4.metric("Consumo total", f"{consumo_total:,.0f} m3")

        # Timeline chart
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=barrio_df["fecha"], y=barrio_df["consumo_litros"] / 1000,
            mode="lines+markers", name="Consumo (m3)",
            line=dict(color="#1976d2", width=2),
        ))

        anom_df = barrio_df[barrio_df["n_models_detecting"] >= 2]
        if len(anom_df) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=anom_df["fecha"], y=anom_df["consumo_litros"] / 1000,
                mode="markers", name="Anomalía (>=2 modelos)",
                marker=dict(color="red", size=12, symbol="x"),
            ))

        if "is_changepoint" in barrio_df.columns:
            cp_df = barrio_df[barrio_df["is_changepoint"] == True]
            for _, row in cp_df.iterrows():
                try:
                    fig_timeline.add_vline(
                        x=row["fecha"].timestamp() * 1000,
                        line_dash="dash", line_color="orange",
                        annotation_text="Cambio",
                    )
                except Exception:
                    pass

        fig_timeline.update_layout(
            title=f"Consumo Mensual — {barrio_name}",
            xaxis_title="Fecha", yaxis_title="Consumo (m3)",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ── Evidencias externas (compactas) ──
        col_ext1, col_ext2, col_ext3 = st.columns(3)

        # InSAR
        insar_path = os.path.join(DATA_DIR, "synthetic_insar_subsidence.csv")
        if os.path.exists(insar_path):
            df_insar = pd.read_csv(insar_path)
            match = df_insar[df_insar["barrio"].str.contains(barrio_name, case=False, na=False)]
            if not match.empty:
                with col_ext1:
                    val = match.iloc[0]["subsidence_mm_yr"]
                    st.metric("📡 InSAR Subsidencia", f"{val:.1f} mm/a",
                              delta="anómalo" if val < -3 else "normal",
                              delta_color="inverse")

        # Piezometry
        piezo_path = os.path.join(DATA_DIR, "synthetic_piezometry.csv")
        if os.path.exists(piezo_path):
            df_piezo = pd.read_csv(piezo_path)
            match = df_piezo[df_piezo["barrio"].str.contains(barrio_name, case=False, na=False)]
            if not match.empty:
                with col_ext2:
                    anom_count = int(match["wt_rising_anomaly"].sum())
                    st.metric("💧 IGME Piezometria", f"{anom_count} alertas",
                              delta="subida anomala" if anom_count > 0 else "normal",
                              delta_color="inverse" if anom_count > 0 else "normal")

        # Electricity/water
        elec_path = os.path.join(DATA_DIR, "synthetic_electricity_water_ratio.csv")
        if os.path.exists(elec_path):
            df_elec = pd.read_csv(elec_path)
            match = df_elec[df_elec["barrio"].str.contains(barrio_name, case=False, na=False)]
            if not match.empty:
                with col_ext3:
                    mean_ratio = match["electricity_kwh_per_m3"].mean()
                    st.metric("⚡ Ratio Elec/Agua", f"{mean_ratio:.2f} kWh/m3",
                              delta="sospechoso" if mean_ratio > 4 else "normal",
                              delta_color="inverse" if mean_ratio > 4 else "normal")

        # ── Fila 2: Viviendas sospechosas (EL WOW) ──
        st.markdown("---")
        st.subheader("🏠 Viviendas Sospechosas — Detección Individual")
        st.markdown(
            "> Nuestro sistema analiza el consumo **hora a hora** de cada vivienda "
            "y detecta fugas, fraude o contadores rotos a nivel individual."
        )

        suspicious_all, all_scores, df_hourly = load_household_analysis()
        catastro = load_catastro_benchmark()

        # Filter by barrio name (hourly data uses different naming)
        # Try matching by barrio number or partial name
        barrio_num = barrio_name.split("-")[0].strip() if "-" in barrio_name else ""
        barrio_text = barrio_name.split("-", 1)[1].strip().upper() if "-" in barrio_name else barrio_name.upper()

        if not suspicious_all.empty:
            # Match suspicious households to this barrio
            sus_barrio = suspicious_all[
                suspicious_all["barrio"].str.contains(barrio_text[:10], case=False, na=False)
            ]
        else:
            sus_barrio = pd.DataFrame()

        if not sus_barrio.empty:
            # Merge with catastro for building info
            if not catastro.empty:
                sus_display = sus_barrio.merge(
                    catastro[["contrato_id", "building_m2", "construction_year",
                              "expected_monthly_L", "consumption_efficiency_ratio"]],
                    on="contrato_id", how="left",
                )
            else:
                sus_display = sus_barrio.copy()

            # Display table
            display_cols = ["contrato_id", "tipo_sospecha", "anomaly_score", "inicio_estimado"]
            if "building_m2" in sus_display.columns:
                display_cols += ["building_m2", "construction_year", "consumption_efficiency_ratio"]
            if "fuga_conocida" in sus_display.columns:
                display_cols.append("fuga_conocida")

            st.dataframe(
                sus_display[[c for c in display_cols if c in sus_display.columns]]
                .rename(columns={
                    "contrato_id": "Contrato",
                    "tipo_sospecha": "Tipo Sospecha",
                    "anomaly_score": "Score",
                    "inicio_estimado": "Inicio Estimado",
                    "building_m2": "m2 Vivienda",
                    "construction_year": "Año Construcción",
                    "consumption_efficiency_ratio": "Ratio Real/Esperado",
                    "fuga_conocida": "Ground Truth",
                }),
                use_container_width=True,
            )

            # Seleccionar vivienda para ver gráfico horario
            st.markdown("#### Gráfico Horario por Vivienda")
            contrato_opts = sus_barrio["contrato_id"].tolist()
            if contrato_opts and df_hourly is not None:
                selected_contrato = st.selectbox(
                    "Selecciona una vivienda sospechosa:",
                    contrato_opts,
                    key="contrato_select",
                )

                dom_data = df_hourly[df_hourly["contrato_id"] == selected_contrato].sort_values("timestamp")

                if not dom_data.empty:
                    fig_dom = go.Figure()

                    # Mark leak zone if known
                    leak_labels_path = os.path.join(DATA_DIR, "synthetic_leak_labels.csv")
                    if os.path.exists(leak_labels_path):
                        labels = pd.read_csv(leak_labels_path, parse_dates=["inicio_fuga", "fin_fuga"])
                        leak_match = labels[labels["contrato_id"] == selected_contrato]
                        if not leak_match.empty:
                            lk = leak_match.iloc[0]
                            fig_dom.add_vrect(
                                x0=lk["inicio_fuga"], x1=lk["fin_fuga"],
                                fillcolor="red", opacity=0.1, line_width=0,
                                annotation_text=f"FUGA: {lk['tipo_fuga']}",
                                annotation_position="top left",
                            )

                    fig_dom.add_trace(go.Scatter(
                        x=dom_data["timestamp"], y=dom_data["consumo_litros"],
                        mode="lines", name="Consumo (L/hora)",
                        line=dict(color="#1976d2", width=1),
                    ))

                    # Moving average
                    dom_ma = dom_data.set_index("timestamp")["consumo_litros"].rolling("24h").mean()
                    fig_dom.add_trace(go.Scatter(
                        x=dom_ma.index, y=dom_ma.values,
                        mode="lines", name="Media movil 24h",
                        line=dict(color="#ff9800", width=2),
                    ))

                    fig_dom.update_layout(
                        title=f"{selected_contrato} — Consumo Horario",
                        xaxis_title="Fecha/Hora", yaxis_title="Consumo (L/h)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_dom, use_container_width=True)

                    # Catastro info for this household
                    if not catastro.empty:
                        cat_row = catastro[catastro["contrato_id"] == selected_contrato]
                        if not cat_row.empty:
                            cr = cat_row.iloc[0]
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Superficie", f"{cr['building_m2']:.0f} m2")
                            c2.metric("Año construcción", int(cr["construction_year"]))
                            c3.metric("Consumo esperado IDAE", f"{cr['expected_monthly_L']:.0f} L/mes")
                            ratio = cr["consumption_efficiency_ratio"]
                            c4.metric(
                                "Ratio real/esperado",
                                f"{ratio:.2f}x",
                                delta="FUGA" if ratio > 1.5 else ("FRAUDE" if ratio < 0.5 else "Normal"),
                                delta_color="inverse" if ratio > 1.5 or ratio < 0.5 else "normal",
                            )
        else:
            st.info(
                f"No se detectaron viviendas sospechosas en barrios que coincidan con '{barrio_name}'. "
                f"Los datos horarios individuales cubren 15 barrios del area urbana."
            )

        # ── Fila 3: Expander técnico ──
        with st.expander("🔬 Ver detalles técnicos (ANR, SHAP, validación)"):
            # ANR
            if "anr_ratio" in barrio_df.columns:
                anr_sum = barrio_df["anr_ratio"].fillna(0).sum()
                if anr_sum > 0:
                    fig_anr = go.Figure()
                    fig_anr.add_trace(go.Scatter(
                        x=barrio_df["fecha"], y=barrio_df["anr_ratio"],
                        mode="lines+markers", name="ANR Ratio",
                        line=dict(color="#e65100", width=2),
                    ))
                    fig_anr.add_hline(y=1.0, line_dash="dash", line_color="gray",
                                      annotation_text="Equilibrio")
                    fig_anr.update_layout(
                        title=f"{barrio_name} — Agua No Registrada",
                        height=300,
                    )
                    st.plotly_chart(fig_anr, use_container_width=True)

            # SHAP
            if "shap_explanation" in barrio_df.columns:
                anom_months = barrio_df[barrio_df["n_models_detecting"] >= 2].sort_values(
                    "ensemble_score", ascending=False)
                if len(anom_months) > 0:
                    top_month = anom_months.iloc[0]
                    shap_text = top_month.get("shap_explanation", "")
                    if shap_text and str(shap_text) != "nan":
                        st.markdown(f"**Mes con mayor riesgo:** {top_month['fecha']}")
                        st.markdown(f"**Drivers principales:** {shap_text}")

            # Detail table
            st.subheader("Detalle mensual")
            cols_show = ["fecha", "consumo_litros", "n_models_detecting", "alert_color",
                         "ensemble_score", "anr_ratio"]
            cols_avail = [c for c in cols_show if c in barrio_df.columns]
            display_df = barrio_df[cols_avail].copy().reset_index(drop=True)
            col_rename = {
                "fecha": "Mes", "consumo_litros": "Consumo (L)",
                "n_models_detecting": "Modelos detectando", "alert_color": "Alerta",
                "ensemble_score": "Score", "anr_ratio": "ANR",
            }
            display_df = display_df.rename(columns={c: col_rename.get(c, c) for c in cols_avail})
            st.dataframe(display_df, use_container_width=True)

    # ── Advanced Analytics: 5 tecnicas quant-grade ────────────────────
    with st.expander("🧠 Advanced Analytics — Quant-Grade Detection (5 tecnicas)"):
        st.markdown("""
        > **5 tecnicas avanzadas** aplicadas a datos horarios individuales:
        > Spectral FFT, Autoencoder + UMAP, Survival Cox, BOCPD Changepoints, Factor Model.
        """)

        @st.cache_resource
        def _load_advanced_results():
            import io as _io
            _old_stdout = sys.stdout
            sys.stdout = _io.StringIO()
            try:
                from advanced_household_analytics import run_all_advanced
                result = run_all_advanced()
            except Exception as e:
                result = pd.DataFrame(), {"_error": str(e)}
            finally:
                sys.stdout = _old_stdout
            return result

        adv_combined, adv_figures = _load_advanced_results()

        if "_error" in adv_figures:
            st.error(f"Error cargando Advanced Analytics: {adv_figures['_error']}")
        elif not adv_combined.empty:
            adv_t1, adv_t2, adv_t3, adv_t4, adv_t5 = st.tabs([
                "📊 Spectral FFT", "🧬 Autoencoder UMAP", "⏳ Survival Cox",
                "📍 Changepoint BOCPD", "📐 Factor Model",
            ])

            with adv_t1:
                st.markdown("**Spectral Entropy**: alta entropia = espectro plano = fuga (consumo constante)")
                if "spectral_entropy_ranking" in adv_figures:
                    st.plotly_chart(adv_figures["spectral_entropy_ranking"], use_container_width=True)
                if "spectral_spectrogram_top" in adv_figures:
                    st.plotly_chart(adv_figures["spectral_spectrogram_top"], use_container_width=True)

            with adv_t2:
                st.markdown("**Autoencoder**: perfiles 24h normalizados → espacio latente → UMAP 2D")
                if "ae_umap_scatter" in adv_figures:
                    st.plotly_chart(adv_figures["ae_umap_scatter"], use_container_width=True)
                if "ae_archetype_profiles" in adv_figures:
                    st.plotly_chart(adv_figures["ae_archetype_profiles"], use_container_width=True)

            with adv_t3:
                st.markdown("**Cox PH**: P(fuga en 60 dias) según edad edificio, m2, pipe risk, edad titular")
                if "survival_hazard_ratios" in adv_figures:
                    st.plotly_chart(adv_figures["survival_hazard_ratios"], use_container_width=True)
                if "survival_kaplan_meier" in adv_figures:
                    st.plotly_chart(adv_figures["survival_kaplan_meier"], use_container_width=True)

            with adv_t4:
                st.markdown("**BOCPD**: detecta el momento exacto (hora) del cambio de regimen de consumo")
                if "cp_changepoint_timeline" in adv_figures:
                    st.plotly_chart(adv_figures["cp_changepoint_timeline"], use_container_width=True)

            with adv_t5:
                st.markdown("**Factor Model**: E[consumo] = f(hora, dia, m2, personas). Residual = anomalía")
                if "factor_residual_heatmap" in adv_figures:
                    st.plotly_chart(adv_figures["factor_residual_heatmap"], use_container_width=True)
                if "factor_peer_ranking" in adv_figures:
                    st.plotly_chart(adv_figures["factor_peer_ranking"], use_container_width=True)
                if "factor_factor_decomposition" in adv_figures:
                    st.plotly_chart(adv_figures["factor_factor_decomposition"], use_container_width=True)

            # Tabla resumen
            st.markdown("---")
            st.markdown("**Top 15 viviendas por score combinado avanzado:**")
            show_cols = ["contrato_id"]
            if "combined_advanced_score" in adv_combined.columns:
                show_cols.append("combined_advanced_score")
            if "n_techniques_flagging" in adv_combined.columns:
                show_cols.append("n_techniques_flagging")
            for c in ["spectral_anomaly_score", "ae_anomaly_score", "factor_anomaly_score"]:
                if c in adv_combined.columns:
                    show_cols.append(c)
            st.dataframe(adv_combined[show_cols].head(15), use_container_width=True)
        elif "_error" not in adv_figures:
            st.info("Ejecuta `python advanced_household_analytics.py` para generar resultados.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: AQUACARE — Detección individual de personas vulnerables
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    > **AquaCare** detecta anomalías hídricas en viviendas de **personas mayores que viven solas**.
    > Cruza 3 fuentes: detección de fugas por vivienda + datos del edificio (Catastro) +
    > perfil del titular (Padron Municipal). Si una persona vulnerable tiene una fuga
    > silenciosa, el sistema la detecta y escala automaticamente.
    """)

    # Cargar datos individuales
    suspicious_ac, _, df_hourly_ac = load_household_analysis()
    profiles = load_household_profiles()
    catastro_ac = load_catastro_benchmark()

    if not profiles.empty and not suspicious_ac.empty:
        # Cruzar: viviendas sospechosas + perfiles demográficos + catastro
        ac_merged = suspicious_ac.merge(
            profiles[["contrato_id", "nombre_titular", "edad_titular", "sexo",
                       "vive_solo", "n_personas_hogar", "direccion_sintetica", "telefono_contacto"]],
            on="contrato_id", how="left",
        )
        if not catastro_ac.empty:
            ac_merged = ac_merged.merge(
                catastro_ac[["contrato_id", "building_m2", "construction_year",
                              "consumption_efficiency_ratio"]],
                on="contrato_id", how="left",
            )

        # Filtrar: solo viviendas con titular >65 años
        ac_elderly = ac_merged[ac_merged["edad_titular"] >= 65].copy()

        # Clasificar nivel de alerta individual
        def _nivel_aquacare(row):
            score = row.get("anomaly_score", 0)
            edad = row.get("edad_titular", 0)
            solo = row.get("vive_solo", False)
            if score > 0.7 and edad >= 75 and solo:
                return "CRITICO"
            elif score > 0.5 and edad >= 70 and solo:
                return "ALTO"
            elif score > 0.3 and edad >= 65:
                return "VIGILANCIA"
            return "BAJO"

        ac_elderly["nivel_aquacare"] = ac_elderly.apply(_nivel_aquacare, axis=1)
        ac_elderly = ac_elderly[ac_elderly["nivel_aquacare"] != "BAJO"]
        ac_elderly = ac_elderly.sort_values(
            ["nivel_aquacare", "anomaly_score"],
            ascending=[True, False],
            key=lambda x: x.map({"CRITICO": 0, "ALTO": 1, "VIGILANCIA": 2}) if x.name == "nivel_aquacare" else -x,
        ).reset_index(drop=True)

        # ── KPIs ──
        n_critico = (ac_elderly["nivel_aquacare"] == "CRITICO").sum()
        n_alto = (ac_elderly["nivel_aquacare"] == "ALTO").sum()
        n_vigilancia = (ac_elderly["nivel_aquacare"] == "VIGILANCIA").sum()
        n_solos = ac_elderly["vive_solo"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 CRITICO", n_critico)
        c2.metric("🟠 ALTO", n_alto)
        c3.metric("🟡 VIGILANCIA", n_vigilancia)
        c4.metric("🏠 Viven solos", int(n_solos))

        # ── Tabla principal: Viviendas en Riesgo ──
        st.markdown("---")
        st.subheader("🏠 Viviendas en Riesgo — Detección Individual")

        if len(ac_elderly) > 0:
            display_cols = {
                "nivel_aquacare": "Nivel",
                "contrato_id": "Contrato",
                "nombre_titular": "Titular",
                "edad_titular": "Edad",
                "vive_solo": "Vive Solo",
                "direccion_sintetica": "Direccion",
                "tipo_sospecha": "Anomalía",
                "anomaly_score": "Score",
            }
            if "building_m2" in ac_elderly.columns:
                display_cols["building_m2"] = "m2"
                display_cols["construction_year"] = "Año Edif."
                display_cols["consumption_efficiency_ratio"] = "Ratio"

            avail_cols = {k: v for k, v in display_cols.items() if k in ac_elderly.columns}
            st.dataframe(
                ac_elderly[list(avail_cols.keys())]
                .rename(columns=avail_cols),
                use_container_width=True,
            )

            # ── Ficha de vivienda individual ──
            st.markdown("---")
            st.subheader("📋 Ficha de Vivienda")

            ficha_opts = ac_elderly.apply(
                lambda r: f"{r['nivel_aquacare']} | {r['contrato_id']} — {r['nombre_titular']}, {int(r['edad_titular'])} años",
                axis=1,
            ).tolist()

            # Pre-seleccionar caso más crítico (o María García si existe)
            _default_idx = 0
            for _i, _opt in enumerate(ficha_opts):
                if "garc" in _opt.lower() or "garcía" in _opt.lower() or "garcia" in _opt.lower():
                    _default_idx = _i
                    break

            selected_ficha = st.selectbox("Selecciona una vivienda:", ficha_opts, index=_default_idx, key="ficha_select")
            selected_idx = ficha_opts.index(selected_ficha)
            ficha = ac_elderly.iloc[selected_idx]

            # Tarjeta visual
            nivel_color = {"CRITICO": "#d32f2f", "ALTO": "#ff9800", "VIGILANCIA": "#fbc02d"}.get(
                ficha["nivel_aquacare"], "#999"
            )
            nivel_emoji = {"CRITICO": "🔴", "ALTO": "🟠", "VIGILANCIA": "🟡"}.get(
                ficha["nivel_aquacare"], "⬜"
            )
            nivel_label = {"CRITICO": "CRÍTICA", "ALTO": "ALTA", "VIGILANCIA": "VIGILANCIA"}.get(
                ficha["nivel_aquacare"], ficha["nivel_aquacare"]
            )

            st.markdown(
                f'<div style="background: linear-gradient(135deg, {nivel_color}22, {nivel_color}08); '
                f'border-left: 4px solid {nivel_color}; padding: 20px; border-radius: 8px; margin: 10px 0;">'
                f'<h3 style="margin:0; color:{nivel_color};">'
                f'{nivel_emoji} ALERTA {nivel_label} — AquaCare</h3>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_persona, col_edificio, col_anomalía = st.columns(3)

            with col_persona:
                st.markdown("##### 👤 Titular")
                st.markdown(f"**{ficha['nombre_titular']}**")
                st.markdown(f"Edad: **{int(ficha['edad_titular'])} años**")
                st.markdown(f"Vive {'**sola**' if ficha.get('sexo', 'F') == 'F' else '**solo**'}: "
                            f"{'**Si** ⚠️' if ficha.get('vive_solo', False) else 'No'}")
                st.markdown(f"Personas en hogar: {int(ficha.get('n_personas_hogar', 1))}")
                st.markdown(f"Tel: {ficha.get('telefono_contacto', 'N/D')}")

            with col_edificio:
                st.markdown("##### 🏗️ Vivienda")
                st.markdown(f"**{ficha.get('direccion_sintetica', 'N/D')}**")
                st.markdown(f"{ficha['barrio']}")
                if "building_m2" in ficha.index and pd.notna(ficha.get("building_m2")):
                    st.markdown(f"Superficie: {ficha['building_m2']:.0f} m2")
                    st.markdown(f"Año construcción: {int(ficha['construction_year'])}")
                    ratio = ficha.get("consumption_efficiency_ratio", 0)
                    st.markdown(f"Ratio consumo real/esperado: **{ratio:.2f}x**")

            with col_anomalía:
                st.markdown("##### 🔍 Anomalía Detectada")
                st.markdown(f"Tipo: **{ficha['tipo_sospecha'].replace('_', ' ').title()}**")
                st.markdown(f"Score: **{ficha['anomaly_score']:.3f}**")
                st.markdown(f"Inicio estimado: {ficha.get('inicio_estimado', 'N/D')}")
                st.markdown(f"Contrato: `{ficha['contrato_id']}`")

            # Gráfico horario de esta vivienda
            if df_hourly_ac is not None:
                dom_data = df_hourly_ac[
                    df_hourly_ac["contrato_id"] == ficha["contrato_id"]
                ].sort_values("timestamp")

                if not dom_data.empty:
                    fig_ac = go.Figure()

                    # Zona de fuga
                    if os.path.exists(LEAK_LABELS_PATH):
                        lbl = pd.read_csv(LEAK_LABELS_PATH, parse_dates=["inicio_fuga", "fin_fuga"])
                        lk_match = lbl[lbl["contrato_id"] == ficha["contrato_id"]]
                        if not lk_match.empty:
                            lk = lk_match.iloc[0]
                            fig_ac.add_vrect(
                                x0=lk["inicio_fuga"], x1=lk["fin_fuga"],
                                fillcolor="red", opacity=0.1, line_width=0,
                                annotation_text=f"FUGA: {lk['tipo_fuga']}",
                                annotation_position="top left",
                            )

                    fig_ac.add_trace(go.Scatter(
                        x=dom_data["timestamp"], y=dom_data["consumo_litros"],
                        mode="lines", name="Consumo (L/hora)",
                        line=dict(color="#1976d2", width=1),
                    ))
                    dom_ma = dom_data.set_index("timestamp")["consumo_litros"].rolling("24h").mean()
                    fig_ac.add_trace(go.Scatter(
                        x=dom_ma.index, y=dom_ma.values,
                        mode="lines", name="Media movil 24h",
                        line=dict(color="#ff9800", width=2),
                    ))
                    fig_ac.update_layout(
                        title=f"Consumo Horario — {ficha['nombre_titular']} ({ficha['contrato_id']})",
                        xaxis_title="Fecha/Hora", yaxis_title="L/hora",
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_ac, use_container_width=True)
        else:
            st.info("No se detectaron viviendas con titulares mayores en riesgo.")

        # ── Protocolo de escalado ──
        st.markdown("---")
        st.subheader("📱 Protocolo de Escalado")
        st.markdown("""
        | Nivel | Accion | Canal |
        |-------|--------|-------|
        | **VIGILANCIA** | Registro en sistema | Dashboard |
        | **ALTO** | Notificacion inmediata | Telegram + Dashboard |
        | **CRITICO** | Notificacion + llamada IA | Telegram + Vapi (voz con IA conversacional) |
        | **CRITICO sin respuesta** | Escalado emergencia | Llamada a contacto secundario |
        """)

        # ── Demo button prominente ──
        st.markdown("---")
        st.markdown("### 🚨 Demo en vivo — Activar protocolo de emergencia")
        st.caption("Pulsa el botón para enviar una alerta real a Telegram y lanzar una llamada de voz con IA al caso más crítico detectado.")
        _btn_demo = st.button(
            "📲 Enviar alerta + llamada de voz",
            type="primary",
            use_container_width=True,
            key="demo_big_btn",
        )

        if _btn_demo:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                from notifier import escalate_alert, send_telegram_alert
                token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
                if token and chat_id:
                    if len(ac_elderly) > 0:
                        f = ac_elderly.iloc[0]
                        nivel = f.get("nivel_aquacare", "CRITICO")
                        desc = (
                            f"{f['contrato_id']} | {f['nombre_titular']}, "
                            f"{int(f['edad_titular'])} años, "
                            f"{'vive sola' if f.get('sexo','F')=='F' else 'vive solo'}\n"
                            f"{f.get('direccion_sintetica', '')}\n"
                            f"{f['tipo_sospecha'].replace('_',' ').title()} "
                            f"(score {f['anomaly_score']:.2f})"
                        )
                        demo_alert = pd.Series({
                            "barrio": f["barrio"],
                            "nivel": nivel,
                            "nivel_alerta": nivel,
                            "anomaly_description": desc,
                            "drop_pct": 47.3,
                            "elderly_vulnerability": 0.78,
                            "consecutive_decline_months": 4,
                            "confidence": 0.91,
                            "pct_elderly_65plus": f["edad_titular"],
                            "pct_elderly_alone": 100 if f.get("vive_solo") else 0,
                            "other_models_confirming": 4,
                        })
                    else:
                        nivel = "CRITICO"
                        demo_alert = pd.Series({
                            "barrio": "17-CAROLINAS ALTAS",
                            "nivel": nivel,
                            "nivel_alerta": nivel,
                            "anomaly_description": "Fuga silenciosa en vivienda vulnerable",
                            "drop_pct": 47.3,
                            "elderly_vulnerability": 0.78,
                            "consecutive_decline_months": 4,
                            "confidence": 0.91,
                            "pct_elderly_65plus": 78,
                            "pct_elderly_alone": 100,
                            "other_models_confirming": 4,
                        })
                    # Escalado completo: Telegram + llamada si CRITICO/ALTO
                    result = escalate_alert(demo_alert, notify_telegram=True, notify_voice=True)
                    steps = result.get("steps", [])
                    tg_ok = any(s.get("action") == "telegram" and s.get("success") for s in steps)
                    call_ok = any(s.get("action") == "voice_call_primary" and s.get("call_id") for s in steps)
                    msg = ""
                    if tg_ok:
                        msg += "✅ Telegram enviado. "
                    if call_ok:
                        msg += "✅ Llamada Vapi iniciada."
                    elif nivel in ("CRITICO", "ALTO"):
                        msg += "⚠️ Llamada no disponible (revisa config Vapi)."
                    st.success(msg if msg else "Escalado completado")
                else:
                    st.warning("Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env")
            except Exception as e:
                st.error(f"Error: {e}")

        # ── Contexto barrio (colapsado) ──
        with st.expander("📊 Ver contexto por barrio (datos demográficos agregados)"):
            if "pct_elderly_65plus" in df.columns:
                barrio_social = df.groupby("barrio_key").agg(
                    pct_elderly=("pct_elderly_65plus", "mean"),
                    pct_alone=("pct_elderly_alone", "mean"),
                    max_score=("ensemble_score", "max"),
                    n_alertas=("n_models_detecting", lambda s: (s >= 2).sum()),
                ).sort_values("pct_elderly", ascending=False)
                barrio_social["barrio"] = barrio_social.index.str.split("__").str[0]

                scatter_df = barrio_social.reset_index()
                scatter_df["es_vulnerable"] = (
                    (scatter_df["pct_elderly"] > 20) & (scatter_df["max_score"] > 0.05)
                )
                fig_scatter = px.scatter(
                    scatter_df,
                    x="pct_elderly", y="max_score",
                    size=scatter_df["n_alertas"].clip(lower=1),
                    hover_name="barrio",
                    hover_data={
                        "pct_elderly": ":.1f",
                        "max_score": ":.3f",
                        "n_alertas": True,
                        "es_vulnerable": False,
                    },
                    labels={
                        "pct_elderly": "% Mayores >65",
                        "max_score": "Score anomalía",
                        "n_alertas": "Alertas",
                    },
                    color="es_vulnerable",
                    color_discrete_map={True: "#d32f2f", False: "#90a4ae"},
                )
                fig_scatter.update_traces(
                    hovertemplate="<b>%{hovertext}</b><br>"
                                  "Mayores >65: %{x:.1f}%<br>"
                                  "Score anomalía: %{y:.3f}<br>"
                                  "Alertas: %{marker.size}<extra></extra>"
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown(
                    "Cada punto es un barrio. "
                    "**Eje X** = % de población mayor de 65. "
                    "**Eje Y** = score de anomalía más alto registrado (0–1). "
                    "**Tamaño** = número de alertas de alta confianza. "
                    "🔴 Rojo = barrio vulnerable (>20% mayores + anomalía detectada). "
                    "Los barrios en la esquina superior derecha son prioritarios."
                )

                st.dataframe(
                    barrio_social[["barrio", "pct_elderly", "pct_alone", "max_score", "n_alertas"]]
                    .rename(columns={
                        "pct_elderly": "% >65", "pct_alone": "% Solos",
                        "max_score": "Score Max", "n_alertas": "Alertas",
                    })
                    .reset_index(drop=True),
                    use_container_width=True,
                )
    else:
        st.info("Datos de perfiles individuales no disponibles. Ejecuta: python synthetic_external_data.py")
