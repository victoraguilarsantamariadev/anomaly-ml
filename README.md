# anomaly-ml

Microservicio Python de detección de anomalías en contadores (agua, electricidad, gas, calor).

Forma parte del proyecto smart-metering. Spring Boot lo llama por HTTP cuando necesita
entrenar un modelo o puntuar lecturas de un contador.

## Estructura

```
anomaly-ml/
├── feature_engineering.py  → cálculo de features (independiente de FastAPI)
├── model.py                → lógica de entrenamiento y scoring (independiente)
├── train_local.py          → script para pruebas locales con CSV
├── main.py                 → servidor FastAPI (microservicio)
├── requirements.txt
├── Dockerfile
├── models/                 → modelos .pkl (uno por contador, gitignored)
└── data/                   → CSV de prueba (gitignored)
```

## Setup

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Pruebas locales con CSV (sin necesitar el backend Java)

```bash
# Con datos de Datadis (tu factura de luz)
python train_local.py --file data/mi_datadis.csv --id mi_contador --mode datadis

# Con datos de Kaggle Smart Meters London
python train_local.py --file data/halfhourly_dataset.csv --id meter_001 --mode kaggle

# Con el CSV del hackathon (datos mensuales por barrio)
python train_local.py --file data/Datos_HACKATHON_AMAEM.csv --mode hackathon

# La gráfica se guarda en output/{meter_id}_anomalies.png
```

## Arrancar el microservicio

```bash
# Desarrollo (con recarga automática al guardar)
uvicorn main:app --reload --port 8001

# Producción
uvicorn main:app --host 0.0.0.0 --port 8001
```

Documentación interactiva en: http://localhost:8001/docs

## Probar los endpoints con curl

```bash
# Health check
curl http://localhost:8001/health

# Entrenar un modelo
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "meter_id": "pp_123",
    "readings": [
      {"timestamp": "2024-01-01T00:00:00", "consumption": 0.1},
      {"timestamp": "2024-01-01T01:00:00", "consumption": 0.05}
    ]
  }'

# Puntuar un punto de datos
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{
    "meter_id": "pp_123",
    "features": {
      "daily_total": 450.0,
      "nocturnal_min": 120.0,
      "nocturnal_mean": 95.0,
      "diurnal_mean": 25.0,
      "night_day_ratio": 3.8,
      "active_hours": 24,
      "is_weekend": 0,
      "zscore": 3.2
    }
  }'
```

## Integración con Docker Compose

```yaml
# En el docker-compose.yml del proyecto smart-metering:
services:
  backend:
    build: ./smart-metering
    ports: ["8080:8080"]

  ml-service:
    build: ./anomaly-ml
    ports: ["8001:8001"]
    volumes:
      - ./anomaly-ml/models:/app/models
```
