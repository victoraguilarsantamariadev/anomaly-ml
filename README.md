# AquaGuard AI

Sistema inteligente de deteccion de anomalias hidricas para Alicante. Detecta fugas, fraude y contadores rotos a nivel de **vivienda individual** y protege a personas mayores vulnerables con alertas automaticas.

**Demo en vivo:** [anomaly-ml-li5iv9vqyaqjusb8xqxkq8.streamlit.app](https://anomaly-ml-li5iv9vqyaqjusb8xqxkq8.streamlit.app)

---

## Que hace

1. **Detecta anomalias** con 16 modelos de ML/estadistica + 5 tecnicas avanzadas (quant-grade)
2. **Cruza 12+ fuentes open source** (Catastro, Sentinel-2, AEMET, INE, Copernicus, IGME...)
3. **Alerta automatica**: Telegram + llamada de voz con IA conversacional en espanol (Vapi)
4. **Protege personas vulnerables**: identifica fugas silenciosas en hogares de mayores que viven solos

---

## Demo rapida

```bash
# Instalar
pip install -r requirements.txt

# Lanzar dashboard
streamlit run dashboard.py

# Ejecutar pipeline completo (16 modelos)
python run_all_models.py

# Ejecutar analytics avanzados (5 tecnicas quant)
python advanced_household_analytics.py

# Probar alertas (Telegram + llamada)
python notifier.py
```

---

## Arquitectura

### 3 niveles de deteccion

| Nivel | Granularidad | Modelos | Datos |
|-------|-------------|---------|-------|
| **Barrio** (mensual) | 42 barrios x 36 meses | IsolationForest, VAE, Autoencoder, Prophet, Chronos, ANR, NMF, Changepoints, TDA, Spatial | 4.3M lecturas (2020-2025) |
| **Sector** (horario) | 23 sectores x 8,760h | Nightflow, ANR, Graph Network | SCADA caudal 2024 (181K) |
| **Vivienda** (horario) | 200 contratos x 1,440h | Spectral FFT, Autoencoder+UMAP, Survival Cox, BOCPD, Factor Model | 288K filas horarias |

### 16 modelos a nivel barrio

| # | Modelo | Tipo | Que detecta |
|---|--------|------|-------------|
| M2 | IsolationForest | ML | Barrios que se comportan raro vs peers |
| M5 | 3-Sigma + IQR | Estadistico | Valores extremos (baseline) |
| M6 | Chronos | Transformer | Desviacion de forecast probabilistico |
| M7 | Prophet | Decomposicion | Fuera de intervalos de confianza |
| M8 | ANR | Ingenieria | Agua inyectada vs facturada |
| M9 | NMF | Caudal nocturno | Consumo 2-5AM / 10-18PM |
| M10 | Meter Readings | Estadistico | Zeros, saltos, lecturas inversas |
| M13 | Autoencoder | Deep Learning | Error de reconstruccion |
| M13-PRO | VAE | Deep Learning | ELBO loss + incertidumbre |
| M14 | Changepoint | PELT + BOCPD | Momento exacto del cambio |
| M14-Net | Graph Network | Topologia | Propagacion en red hidraulica |
| M16 | TDA | Topologico | Homologia persistente |
| M11 | Spatial | Geoestadistico | Cluster vs aislado (Moran's I) |
| M12 | Fraud XGBoost | PU Learning | Meta-modelo con 190 fraudes reales |
| -- | Causal Forest | Causal | Efectos heterogeneos por barrio |
| -- | Transfer Entropy | Info Theory | Flujo de informacion entre barrios |

### 5 tecnicas quant-grade a nivel vivienda

| Tecnica | Concepto | Precision@10 |
|---------|----------|-------------|
| **Spectral FFT** | Entropia espectral — fuga = espectro plano | 100% |
| **Autoencoder + UMAP** | Arquetipos de consumo 24h, clusters anomalos | 100% |
| **Survival Cox PH** | P(fuga en 60 dias) segun edad edificio, m2, etc. | - |
| **BOCPD** | Hora exacta del cambio de regimen (Adams & MacKay 2007) | - |
| **Factor Model** | E[consumo] = f(hora, dia, m2, personas). Residual = anomalia | 100% |

**Precision@10 combinado = 100%** (las 10 primeras viviendas son todas fugas reales)

### Ensemble inteligente

- **Weighted Voting**: pesos del ablation study (contribucion marginal)
- **Stacking**: GBM meta-learner con walk-forward CV
- **Conformal Prediction**: p-valores calibrados (alpha=0.05)
- **SHAP**: explicacion de cada deteccion

---

## Datos Open Source (12+ fuentes)

| Fuente | Datos | Para que |
|--------|-------|---------|
| **AMAEM** | 4.3M lecturas, caudal SCADA, contadores | Consumo base |
| **Catastro INSPIRE WFS** (DGC) | 1,688 edificios con ano construccion | Pipe risk: edificio viejo = tuberia vieja |
| **Padron Municipal** (INE) | Poblacion >65 por barrio | Vulnerabilidad social |
| **Sentinel-2 NDVI** (Copernicus/ESA) | Vegetacion por barrio | Riego excesivo detectado via satelite |
| **AEMET OpenData** | Temperatura y precipitacion | Ajustar consumo esperado por clima |
| **Registro Turismo** (Generalitat Valenciana) | 3,334 viviendas turisticas | Picos estacionales |
| **INE Atlas Renta** | Renta por seccion censal | Fraude por necesidad vs picaresca |
| **InSAR EGMS** (Copernicus) | Subsidencia del terreno mm/ano | Hundimiento = fuga subterranea |
| **Landsat Thermal** (USGS/NASA) | Temperatura de superficie | Cold spots = agua filtrandose |
| **IGME-SINAS** | Nivel freatico (piezometria) | Nivel sube sin lluvia = fuga inyectando |
| **REE** | Consumo electrico por zona | Ratio kWh/m3 alto = bomba ilegal |
| **IDAE** | 128 L/persona/dia benchmark | Consumo esperado por vivienda |

---

## AquaCare — Proteccion de personas vulnerables

El sistema cruza deteccion de fugas + datos del edificio (Catastro) + perfil del titular (Padron) para identificar situaciones criticas:

> **CTR-17-00070** | Maria Garcia Lopez, 78 anos, vive sola
> C/ Azorin 14, 2o B | Edificio 1972, 68m2
> Fuga silenciosa (score 0.82) | Consumo 1.7x esperado
> **NIVEL: CRITICO** — Verificacion presencial inmediata

### Protocolo de escalado

| Nivel | Accion | Canal |
|-------|--------|-------|
| VIGILANCIA | Registro en sistema | Dashboard |
| ALTO | Notificacion inmediata | Telegram + Dashboard |
| CRITICO | Notificacion + llamada IA | Telegram + Vapi (voz en espanol) |
| CRITICO sin respuesta | Escalado emergencia | Llamada a contacto secundario |

---

## Dashboard

3 pestanas interactivas:

1. **Mapa de Riesgo** — Vista ejecutiva con mapa de Alicante + KPIs
2. **Investigar Barrio** — Drilldown a vivienda individual + 5 tecnicas avanzadas
3. **AquaCare** — Viviendas vulnerables + ficha individual + boton de alerta en vivo

---

## Validacion

- **22 capas de validacion** incluyendo null permutation test (p=0.001)
- **Out-of-sample 2025**: rho=+0.63, p=0.027 — anomalias confirmadas en datos no vistos
- **Correlacion con fraude real**: r=+0.54 con cambios de contador
- **4.3M lecturas individuales**: rho=+0.79, q_BH=0.016
- **Fisher's combined test**: p=0.002

---

## Configuracion alertas (.env)

```env
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
VAPI_API_KEY=tu_api_key
VAPI_PHONE_NUMBER_ID=tu_phone_id
CONTACT_PHONE_NUMBER=+34XXXXXXXXX
```

---

## Stack

Python 3.12 | Streamlit | Plotly | scikit-learn | lifelines | UMAP | Vapi (voz IA) | Telegram Bot API

---

**Datathon AMAEM 2024 — Alicante**
