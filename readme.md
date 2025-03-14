# AUTOML Zones Extraction Scripts and Guide

Step-by-Step Explanation
Data Parsing: Read gene and transcript data from files, extracting relevant genomic coordinates and nucleotide sequences.

Transition Extraction:

- EI (Exon → Intron): Validate 'gt' splice site, extract 5 nucleotides left and 7 right of intron start.

- IE (Intron → Exon): Validate 'ag' splice site, extract 100 nucleotides left and 5 right of exon start.

- ZE (Intergenic → First Exon): Extract 500 nucleotides left and 50 right of the first exon's start.

- EZ (Last Exon → Intergenic): Extract 50 nucleotides left and 500 right of the last exon's end.

- False Examples: Generate random nucleotide sequences for each transition type when valid extractions aren't possible.

## Data Storage: Save results into CSV files with each nucleotide in separate columns.

### Solution Code

data_ensembl/
    (contains raw data files)
data/
    data_ei.csv
    data_ie.csv
    data_ze.csv
    data_ez.csv
main_extractor.py

#### Data

Gene:
([GEN_ID],[Cord_inicio_gen],[Cord_final_gen],[string_nucleotides],[chromosome_number],[Cord_global_inicio_gen],[Cord_global_final_gen],strand)
example:
([ENSG00000157005.4],[1000],[2482],[...],[3],[187667913],[187671395],false)

Transcript Information Lines:
([transcript_exon1_start, transcript_exon1_end],[transcript_exon2_start, transcript_exon2_end],...,[transcript_exonN_start, transcript_exonN_end],[transcript_count])

Example:
([1000,1240],[2117,2482],[1])

Detailed Extraction Process
The extraction logic implemented in init.ipynb performs the following steps for each gene that contains transcript data:

1. Exon → Intron (EI)
Locate Transition:
Identify the end position of the first exon (e.g., 1240).

Determine Intron Start:
The intron is assumed to start at exon_end + 1 (i.e., 1241).

Validation:
Confirm that the intron starts with the nucleotides "gt", which should be located at positions 1241–1242.

Extraction:

Extract 5 characters immediately to the left of the intron start.
Extract 5 characters immediately to the right of the intron start.
Concatenate these substrings to form a 12-character sequence.
Storage:
Save the result into data_ei.csv. Each character of the sequence is stored in its own column (B1, B2, …, B12).

2. Intron → Exon (IE)
Locate Transition:
Identify the start position of the second exon (e.g., 2117).

Determine Intron End:
The intron is assumed to end at exon_start - 1 (i.e., 2116).

Validation:
Confirm that the intron ends with "ag", found at positions 2115–2116.

Extraction:

Extract 100 characters immediately to the left of the exon start.
Extract 5 characters immediately to the right of the exon start.
Concatenate these to form a 105-character sequence.
Storage:
Save the result into data_ie.csv, with each character in a separate column (B1 to B105).

3. Intergenic Zone → First Exon (ZE)
Locate Transition:
Use the start position of the first exon (e.g., 1000).

Extraction:

Extract 500 characters immediately to the left of the first exon.
Extract 50 characters immediately to the right of the first exon.
Concatenate these to form a 550-character sequence.
Storage:
Save the result into data_ze.csv, with each character occupying its own column (B1 to B550).

4. Last Exon → Intergenic Zone (EZ)
Locate Transition:
Use the end position of the last exon (e.g., 2482).

Extraction:

Extract 50 characters immediately to the left of the last exon.
Extract 500 characters immediately to the right of the last exon.
Concatenate these to form a 550-character sequence.
Storage:
Save the result into data_ez.csv, with each character in its respective column (B1 to B550).

Files and Outputs
After running the init.ipynb notebook, the following CSV files are generated in the designated data folder:

data_ei.csv: Contains the EI transition sequences.
data_ie.csv: Contains the IE transition sequences.
data_ze.csv: Contains the ZE transition sequences.
data_ez.csv: Contains the EZ transition sequences.
Each CSV file includes metadata (such as gene ID, chromosome number, and genomic coordinates) along with the extracted transition sequence distributed across multiple columns.

# Modelos de Aprendizaje Automático

Además del proceso de extracción de datos, este proyecto incluye la implementación de modelos de machine learning para la predicción de zonas de transición genómica. Se han desarrollado modelos para dos tipos de transiciones:

## Tipos de Modelos Implementados

1. **Modelo EI (Exón → Intrón)**
   - Predice si una secuencia de 12 nucleótidos representa una transición válida de exón a intrón.
   - Utiliza una arquitectura CNN (Red Neuronal Convolucional) para capturar patrones locales en la secuencia.
   - Reconoce el patrón canónico "GT" en el sitio de empalme donador.

2. **Modelo IE (Intrón → Exón)**
   - Predice si una secuencia de 105 nucleótidos representa una transición válida de intrón a exón.
   - Implementa una arquitectura LSTM Bidireccional para capturar dependencias a largo plazo.
   - Identifica el patrón canónico "AG" en el sitio de empalme aceptor.

## Arquitecturas Disponibles

Para cada tipo de transición, se han implementado tres arquitecturas diferentes:

1. **CNN (Red Neuronal Convolucional)**
   - Efectiva para detectar motivos locales en sitios de empalme.
   - Múltiples capas convolucionales 1D con pooling y dropout.

2. **LSTM Bidireccional**
   - Captura dependencias a largo plazo en la secuencia.
   - Adecuada para secuencias largas como las de IE.

3. **Modelo Híbrido (CNN+LSTM)**
   - Combina las fortalezas de ambas arquitecturas.
   - Una rama CNN para patrones locales y una rama LSTM para dependencias a largo plazo.

## Uso de los Modelos

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Entrenamiento de Modelos

Para entrenar los modelos con los datos extraídos:

```bash
python train_models.py
```

Este script entrenará los modelos EI y IE y guardará los mejores modelos en archivos .h5.

### Predicción

Para realizar predicciones con los modelos entrenados:

```bash
python predict.py
```

También puedes procesar secuencias en formato FASTA modificando el script `predict.py`.

### Personalización

En los scripts `train_models.py` y `predict.py` puedes:

- Cambiar las arquitecturas descomentando/comentando las líneas correspondientes.
- Ajustar hiperparámetros como tasas de aprendizaje, tamaños de batch, etc.
- Modificar las funciones de pérdida o métricas según tus necesidades.

## Métricas de Evaluación

Los modelos se evalúan utilizando:

- Exactitud (Accuracy)
- Sensibilidad (Recall para la clase positiva)
- Especificidad (Recall para la clase negativa)
- Área bajo la curva ROC (AUC)

## Extensión a Otros Tipos de Transiciones

Los modelos implementados para EI e IE pueden adaptarse fácilmente para las transiciones ZE y EZ siguiendo la misma estructura, pero ajustando los tamaños de entrada según las dimensiones de las secuencias correspondientes (550 nucleótidos).

# Automatización y Flujo de Trabajo Completo

Este proyecto ahora incluye un sistema completo de automatización para todo el flujo de trabajo, desde la extracción de datos hasta la evaluación de modelos, incluyendo un sistema de ensemble para combinar diferentes modelos.

## Nuevos Scripts Implementados

1. **ensemble_predict.py**: Implementa un sistema de ensemble que permite:
   - Cargar y utilizar todos los modelos disponibles
   - Realizar predicciones automáticas basadas en la longitud de la secuencia
   - Evaluar el rendimiento de todos los modelos en datos de prueba
   - Generar visualizaciones comparativas
   - Procesar lotes de secuencias

2. **run_automl.py**: Script principal que orquesta todo el flujo de trabajo:
   - Extracción de datos a partir de archivos genómicos
   - Entrenamiento de todos los modelos
   - Evaluación y visualización de resultados
   - Generación de informes y gráficos

## Ejecución del Proyecto Completo

Para ejecutar todo el flujo de trabajo:

```bash
python run_automl.py --all
```

También puedes ejecutar cada fase de forma independiente:

```bash
# Solo extracción de datos
python run_automl.py --extract --input_file ./data_ensembl/tu_archivo.txt

# Solo entrenamiento de modelos
python run_automl.py --train

# Solo evaluación de modelos
python run_automl.py --evaluate
```

## Visualizaciones y Análisis

El sistema genera automáticamente varias visualizaciones:

1. **Distribuciones de Probabilidad**: Muestra cómo se distribuyen las probabilidades para ejemplos positivos y negativos en cada tipo de transición.

2. **Comparación de Rendimiento**: Gráficos de barras y heatmaps que comparan las métricas de rendimiento entre los diferentes modelos.

3. **Historiales de Entrenamiento**: Visualización de la evolución de la precisión y pérdida durante el entrenamiento.

## Uso del Sistema de Ensemble

Para utilizar el sistema de ensemble en tus propios datos:

```python
from ensemble_predict import GeneticTransitionEnsemble

# Inicializar el ensemble
ensemble = GeneticTransitionEnsemble()

# Realizar predicción individual
sequence = "aagctGTaagct"  # Secuencia EI
result = ensemble.predict(sequence)
print(f"Predicción: {result['results']['prediction']}")
print(f"Probabilidad: {result['results']['probability']:.4f}")

# Procesar un dataset completo
import pandas as pd
sequences_df = pd.DataFrame({
    'sequence': ["aagctGTaagct", "c"*100 + "agAAA"]
})
results_df = ensemble.batch_predict(sequences_df, 'sequence')
print(results_df)
```

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

1. **Nuevos tipos de transiciones**: Puedes añadir nuevos extractores y modelos siguiendo el mismo patrón.

2. **Nuevas arquitecturas**: Puedes implementar arquitecturas adicionales en `train_models.py`.

3. **Flujos de trabajo personalizados**: Puedes modificar `run_automl.py` para adaptarlo a tus necesidades específicas.
