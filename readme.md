# Proyecto AutoML para Predicción de Estructuras Genéticas

## Descripción
Este proyecto utiliza técnicas de AutoML y Deep Learning para la predicción de estructuras genéticas. Implementa un flujo de trabajo automatizado para entrenar, evaluar y comparar diversos modelos, presentando los resultados en un leaderboard interactivo.

## Características principales
- Implementación de múltiples frameworks de AutoML
- Comparativa detallada de todos los modelos generados
- Dashboard con indicadores de rendimiento
- Flujo de trabajo automatizado para procesamiento y modelado
- Seguimiento y gestión de experimentos con MLFlow
- Optimización de hiperparámetros

## Estructura del proyecto
```
.
├── data/                  # Datos de entrada y procesados
├── notebooks/             # Jupyter notebooks para exploración
├── src/                   # Código fuente del proyecto
│   ├── preprocessing/     # Preprocesamiento de datos genéticos
│   ├── models/            # Implementación de modelos AutoML
│   ├── evaluation/        # Métricas y evaluación
│   └── visualization/     # Generación de gráficos y dashboard
├── mlflow/                # Artefactos y registros de MLFlow
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Esta documentación
```

## Instalación
1. Clonar el repositorio
2. Crear entorno virtual: `python -m venv venv`
3. Activar entorno: `source venv/Scripts/activate` (Windows) o `source venv/bin/activate` (Unix)
4. Instalar dependencias: `pip install -r requirements.txt`

## Compatibilidad
Este proyecto está optimizado para funcionar en Windows, eliminando dependencias problemáticas como auto-sklearn y configurando alternativas compatibles.

## Flujo de trabajo
1. Preprocesamiento de datos genéticos
2. Entrenamiento automatizado con múltiples frameworks AutoML
3. Evaluación y comparación de modelos
4. Visualización de resultados
5. Despliegue del mejor modelo con MLFlow

## Frameworks de AutoML utilizados
- TPOT - Optimización de tuberías basada en programación genética
- LazyPredict - Evaluación rápida de múltiples algoritmos
- PyCaret - Framework de ML de bajo código
- FLAML - AutoML eficiente y económico
- AutoKeras - AutoML basado en Keras (Deep Learning) 