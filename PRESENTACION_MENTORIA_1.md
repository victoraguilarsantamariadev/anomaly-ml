# Resumen — 1a Mentoria Hackathon Agua Alicante

## Que presentamos

Presentamos **AquaGuard AI**, un sistema de inteligencia artificial que analiza los datos de consumo de agua de Alicante para detectar **anomalias hidricas**: situaciones donde el agua se comporta de forma rara en un barrio. Eso incluye:

- **Fugas de agua** en la red de tuberias (el agua se pierde antes de llegar a las casas)
- **Fraude o manipulacion de contadores** (alguien altera su contador para pagar menos)
- **Averias en contadores** (el contador marca mal: ceros, negativos, lecturas imposibles)
- **Perdidas silenciosas** (una fuga pequeña que nadie nota pero que acumula miles de litros)

En resumen: cualquier situacion donde el agua "desaparece" o los numeros no cuadran.

### Datos que usamos

Trabajamos con **9 fuentes de datos**, combinando datos del hackathon con datos publicos open data:

- **Del hackathon (AMAEM):** 4.3 millones de lecturas de contadores, consumo mensual de 42 barrios durante 6 anos (2020-2025), cambios de contadores, flujo nocturno, red de tuberias y agua regenerada
- **Open data (9 fuentes):**
  - Padron Municipal de Alicante 2025 (censo: edad, personas que viven solas)
  - AEMET (temperatura y precipitacion oficial)
  - INE ocupacion hotelera (turismo mensual)
  - SPEI/CSIC (indice de sequia)
  - Calendario de festivos de Alicante
  - **Sentinel-2 (ESA)** — imagenes de satelite para detectar zonas sospechosamente verdes en sequia
  - **Inside Airbnb** — densidad de pisos turisticos para separar anomalias por turismo de las reales
  - **INE Atlas de Renta** — renta media por barrio para distinguir fraude por necesidad vs codicia
  - **Catastro (DGC)** — edad de edificios para predecir fugas por tuberias viejas

### Como funciona

Probamos **14 modelos de deteccion** diferentes (cada uno busca anomalias de una forma distinta). Tras pruebas rigurosas, **solo 6 demostraron que aportan valor real** — los otros 8 los descartamos porque no mejoraban o empeoraban los resultados.

Los 6 modelos activos funcionan como un "jurado": cada uno vota si un barrio tiene algo raro. Cuando varios coinciden, la alerta es mas fiable.

### Validacion

Tenemos **22 pruebas independientes** de que las detecciones son reales y no inventadas. Las mas fuertes:
- Coincidencia del 79% con lecturas reales de contadores
- Predice correctamente anomalias en datos de 2025 que nunca habia visto
- Tres fuentes fisicas independientes confirman los resultados (p = 0.002)
- 1000 pruebas aleatorias: ninguna iguala al sistema real

### AquaCare (componente social)

Cruzamos las anomalias detectadas con el censo de poblacion para identificar barrios donde viven **personas mayores de 65 anos que viven solas** — las mas vulnerables ante una fuga silenciosa o un problema con el agua que no saben resolver.

### Que entregamos

Un dashboard interactivo de 8 paginas, un informe HTML de 133 KB con 13 secciones, y todo el codigo abierto y reproducible.

---

## Feedback del mentor

El consejo principal fue:

> **Adaptar TODO mucho mas a un publico no tecnico.**

En concreto:

1. **Explicar que es una anomalia** — No dar por hecho que el jurado sabe lo que es. Decir claramente: "una anomalia es una fuga, un fraude, un contador roto, agua que desaparece". Poner ejemplos reales y tangibles.

2. **Traducir cada metrica y numero** — El jurado no sabe que es un p-value, una correlacion, o un AUC. Cada dato tiene que ir acompañado de una frase que lo explique. En vez de "rho = 0.79, p = 0.003", decir "coincidimos con los datos reales de contadores el 79% de las veces, y la probabilidad de que sea casualidad es de 1 entre 333".

3. **Que un nino de 10 anos lo entienda** — Si el jurado tiene que pensar para entender un grafico o un numero, ya hemos perdido. Cada visualizacion necesita un titulo claro y una frase explicativa debajo.

4. **El impacto emocional importa** — No es solo tecnico. Hablar de las personas: los mayores solos con fugas que no detectan, el agua que se pierde en una ciudad con sequia, el dinero publico que se ahorra.

---

## Acciones tomadas tras la mentoria

- [x] Anadir 4 fuentes de datos externas creativas (NDVI satelite, Airbnb, Renta INE, Catastro)
- [x] Nueva pagina en dashboard: "Datos Externos" con 4 secciones + indice combinado
- [x] Integrar features creativas en el pipeline (run_all_models.py)
- [ ] Anadir explicaciones no tecnicas a cada pagina del dashboard
- [ ] Explicar que es una anomalia (fugas, fraude, averias) en la intro de cada seccion
- [ ] Traducir todas las metricas a lenguaje normal
- [ ] Arreglar la pagina AquaCare del dashboard (actualmente muestra datos vacios)
- [ ] Preparar el pitch oral aplicando el mismo criterio de simplicidad
