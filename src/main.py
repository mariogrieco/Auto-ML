import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lazypredict.Supervised import LazyClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random

# Importar el extractor
import sys
sys.path.append('.')
from main_extractor import Extraction

# Configuración de MLflow
mlflow.set_tracking_uri("./mlflow")
mlflow.set_experiment("automl-genetic-transition-prediction")

# Configuración de estilo para las gráficas
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def extract_genetic_data(input_file_path, output_dir="./data"):
    """Extrae datos de transiciones genéticas usando el Extraction"""
    print(f"Extrayendo datos genéticos desde {input_file_path}")
    extractor = Extraction(input_file_path, output_path=output_dir)
    extractor.process_file()
    extractor.save_to_csv()
    print("Extracción de datos completada.")
    return {
        'ei': f"{output_dir}/data_ei.csv",
        'ei_random': f"{output_dir}/data_ei_random.csv",
        'ie': f"{output_dir}/data_ie.csv",
        'ie_random': f"{output_dir}/data_ie_random.csv",
        'ze': f"{output_dir}/data_ze.csv",
        'ze_random': f"{output_dir}/data_ze_random.csv",
        'ez': f"{output_dir}/data_ez.csv",
        'ez_random': f"{output_dir}/data_ez_random.csv"
    }

def prepare_transition_data(true_data_path, false_data_path, min_samples=20):
    """Prepara los datos para clasificación de transiciones"""
    print(f"Preparando datos de transición desde {true_data_path} y {false_data_path}")
    
    # Cargar datos verdaderos y falsos
    true_df = pd.read_csv(true_data_path)
    false_df = pd.read_csv(false_data_path)
    
    # Verificar si hay suficientes muestras
    print(f"Muestras positivas: {len(true_df)}, Muestras negativas: {len(false_df)}")
    
    # Si hay muy pocas muestras, generar datos sintéticos
    if len(true_df) < min_samples or len(false_df) < min_samples:
        print("Insuficientes muestras reales. Generando datos sintéticos adicionales...")
        
        # Identificar columnas de características
        feature_cols = [col for col in true_df.columns if col.startswith('B')]
        
        if len(feature_cols) == 0:
            print("No se encontraron columnas de características que empiecen con 'B'")
            return None, None
        
        # Generar muestras positivas adicionales
        if len(true_df) > 0 and len(true_df) < min_samples:
            # Usar la primera muestra como plantilla
            template_row = true_df.iloc[0]
            
            # Crear nuevas filas con pequeñas variaciones en las características
            new_rows = []
            for i in range(min_samples - len(true_df)):
                new_row = template_row.copy()
                
                # Modificar algunas características aleatoriamente (solo nucleótidos)
                for col in feature_cols:
                    if random.random() < 0.3:  # 30% de probabilidad de cambio
                        new_row[col] = random.choice(['a', 'c', 'g', 't'])
                
                new_rows.append(new_row)
            
            # Añadir las nuevas filas al DataFrame
            synthetic_true_df = pd.DataFrame(new_rows)
            true_df = pd.concat([true_df, synthetic_true_df], ignore_index=True)
        
        # Generar muestras negativas adicionales
        if len(false_df) > 0 and len(false_df) < min_samples:
            # Usar la primera muestra como plantilla
            template_row = false_df.iloc[0]
            
            # Crear nuevas filas con pequeñas variaciones
            new_rows = []
            for i in range(min_samples - len(false_df)):
                new_row = template_row.copy()
                
                # Modificar algunas características aleatoriamente
                for col in feature_cols:
                    if random.random() < 0.3:
                        new_row[col] = random.choice(['a', 'c', 'g', 't'])
                
                new_rows.append(new_row)
            
            # Añadir las nuevas filas al DataFrame
            synthetic_false_df = pd.DataFrame(new_rows)
            false_df = pd.concat([false_df, synthetic_false_df], ignore_index=True)
    
    # Añadir etiquetas (1 para verdaderos, 0 para falsos)
    true_df['target'] = 1
    false_df['target'] = 0
    
    # Combinar los datos
    combined_df = pd.concat([true_df, false_df], ignore_index=True)
    
    # Separar características y objetivo
    feature_cols = [col for col in combined_df.columns if col.startswith('B')]
    X = combined_df[feature_cols]
    y = combined_df['target']
    
    print(f"Dataset final: {len(X)} muestras, {len(feature_cols)} características")
    
    return X, y

def train_model_for_transition(transition_type, X, y):
    """Entrena modelos para un tipo específico de transición"""
    print(f"\nEntrenando modelos para transición {transition_type}...")
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Usar LazyPredict para entrenar múltiples modelos
    with mlflow.start_run(run_name=f"lazypredict-{transition_type}"):
        lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        try:
            models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
            
            # Guardar resultados
            models.to_csv(f'results/lazypredict_{transition_type}.csv')
            
            # Obtener el mejor modelo
            best_model_name = models.index[0]
            best_model = lazy_clf.models[best_model_name]
            
            # Evaluar el mejor modelo en conjunto de prueba
            y_pred = best_model.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Guardar el mejor modelo
            model_path = f'models/{transition_type}_best_model.joblib'
            joblib.dump(best_model, model_path)
            
            # Registrar en MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_params({"best_model": best_model_name})
            mlflow.sklearn.log_model(best_model, f"{transition_type}_model")
            
            return {
                'transition_type': transition_type,
                'best_model_name': best_model_name,
                'metrics': metrics,
                'all_models': models,
                'model_path': model_path
            }
            
        except Exception as e:
            print(f"Error al entrenar modelos para {transition_type}: {str(e)}")
            return {
                'transition_type': transition_type,
                'error': str(e)
            }

def create_unified_leaderboard(results):
    """Crea un leaderboard unificado para todos los tipos de transición"""
    print("\nCreando leaderboard unificado...")
    
    leaderboard_data = []
    
    for result in results:
        if 'metrics' in result:
            entry = {
                'Transition': result['transition_type'],
                'Best Model': result['best_model_name'],
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1': result['metrics']['f1']
            }
            leaderboard_data.append(entry)
    
    # Crear DataFrame del leaderboard
    leaderboard = pd.DataFrame(leaderboard_data)
    
    # Ordenar por F1-score
    if 'F1' in leaderboard.columns and not leaderboard.empty:
        leaderboard = leaderboard.sort_values('F1', ascending=False)
    
    # Guardar leaderboard
    leaderboard.to_csv("results/unified_leaderboard.csv", index=False)
    
    return leaderboard

def create_visualizations(leaderboard, results):
    """Crea visualizaciones comparativas mejoradas para todos los tipos de transición"""
    print("\nGenerando visualizaciones...")
    
    # Crear directorio para las gráficas si no existe
    Path("visualizations").mkdir(exist_ok=True)
    
    # 1. Gráfica de barras agrupadas para comparar métricas entre tipos de transición
    plt.figure(figsize=(14, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(leaderboard['Transition']))
    width = 0.2
    
    # Colores para cada métrica
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(metrics)))
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, leaderboard[metric], width, label=metric, color=colors[i], alpha=0.8)
    
    plt.xlabel('Tipo de Transición', fontsize=12)
    plt.ylabel('Puntuación', fontsize=12)
    plt.title('Comparación de Métricas por Tipo de Transición Genómica', fontsize=16, fontweight='bold')
    plt.xticks(x + width*1.5, leaderboard['Transition'], rotation=45, fontsize=11)
    plt.ylim(0, 1.05)  # Asegurar que el eje Y va de 0 a 1.05
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('visualizations/transition_metrics_comparison.png', dpi=300)
    plt.close()
    
    # 2. Gráfica de barras para los mejores modelos con etiquetas y colores
    plt.figure(figsize=(14, 8))
    bar_plot = sns.barplot(
        data=leaderboard, 
        x='Transition', 
        y='F1', 
        palette='viridis',
        hue='Best Model'
    )
    
    # Añadir etiquetas a las barras
    for i, v in enumerate(leaderboard['F1']):
        plt.text(i, v+0.02, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')
    
    # Añadir nombres de modelos sobre cada barra
    for i, model in enumerate(leaderboard['Best Model']):
        plt.text(i, 0.05, model, ha='center', rotation=90, fontsize=9, color='white')
    
    plt.title('Rendimiento de Modelos por Tipo de Transición (F1-Score)', fontsize=16, fontweight='bold')
    plt.xlabel('Tipo de Transición', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.xticks(rotation=45, fontsize=11)
    plt.ylim(0, 1.1)  # Asegurar que el eje Y va de 0 a 1.1
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/best_models_by_transition.png', dpi=300)
    plt.close()
    
    # 3. Gráfica comparativa de todos los modelos a través de todas las transiciones
    # Recopilar datos para todos los modelos en todas las transiciones
    all_models_data = []
    common_models = set()
    
    # Encontrar modelos comunes entre todas las transiciones
    for result in results:
        if 'all_models' in result:
            models_df = result['all_models']
            common_models.update(set(models_df.index[:5]))  # Tomar los 5 mejores modelos
    
    common_models = list(common_models)
    
    # Recopilar datos para estos modelos en cada transición
    for result in results:
        if 'all_models' in result:
            transition = result['transition_type']
            models_df = result['all_models']
            
            for model in common_models:
                if model in models_df.index:
                    all_models_data.append({
                        'Transition': transition,
                        'Model': model,
                        'Accuracy': models_df.loc[model, 'Accuracy'] if 'Accuracy' in models_df.columns else 0
                    })
    
    if all_models_data:
        all_models_df = pd.DataFrame(all_models_data)
        
        plt.figure(figsize=(16, 10))
        sns.barplot(
            data=all_models_df, 
            x='Model', 
            y='Accuracy',
            hue='Transition',
            palette='viridis'
        )
        plt.title('Rendimiento de Modelos Comunes a través de Tipos de Transición', fontsize=16, fontweight='bold')
        plt.xlabel('Modelo', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.legend(title='Transición', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/models_comparison_across_transitions.png', dpi=300)
        plt.close()
    
    # 4. Gráficas individuales detalladas de los modelos para cada transición
    for result in results:
        if 'all_models' in result:
            transition_type = result['transition_type']
            models_df = result['all_models'].head(10)  # Top 10 modelos
            
            # Convertir el índice a columna 'Model'
            plot_df = models_df.reset_index()
            plot_df.rename(columns={'index': 'Model'}, inplace=True)
            
            if 'Accuracy' in plot_df.columns and len(plot_df) > 0:
                plt.figure(figsize=(12, 8))
                bars = sns.barplot(data=plot_df, y='Model', x='Accuracy', palette='viridis')
                
                # Añadir etiquetas de valores
                for i, v in enumerate(plot_df['Accuracy']):
                    plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)
                
                plt.title(f'Top 10 Modelos para Transición {transition_type.upper()}', fontsize=16, fontweight='bold')
                plt.xlabel('Accuracy', fontsize=12)
                plt.ylabel('Modelo', fontsize=12)
                plt.xlim(0, 1.05)
                plt.grid(axis='x', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'visualizations/{transition_type}_top_models.png', dpi=300)
                plt.close()
    
    # 5. Gráfica de radar para comparar las métricas de los mejores modelos
    if len(leaderboard) >= 2:  # Solo si hay al menos 2 transiciones para comparar
        plt.figure(figsize=(10, 10))
        
        # Preparar datos para gráfico de radar
        categories = metrics
        N = len(categories)
        
        # Crear ángulos para el gráfico de radar
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Cerrar el círculo
        
        # Inicializar el subplot
        ax = plt.subplot(111, polar=True)
        
        # Dibujar cada tipo de transición
        for i, row in leaderboard.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Cerrar el círculo
            
            # Dibujar la línea y rellenar el área
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Transition'])
            ax.fill(angles, values, alpha=0.1)
        
        # Fijar etiquetas
        plt.xticks(angles[:-1], categories, fontsize=12)
        
        # Dibujar líneas radiales para cada ángulo
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], fontsize=10)
        plt.ylim(0, 1)
        
        # Añadir leyenda y título
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Métricas Comparativas por Tipo de Transición', fontsize=16, fontweight='bold', y=1.08)
        
        plt.tight_layout()
        plt.savefig('visualizations/radar_metrics_comparison.png', dpi=300)
        plt.close()
    
    print("Visualizaciones mejoradas guardadas en el directorio 'visualizations'")

def main():
    # Configurar directorios
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("mlflow", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Extraer datos genéticos si no existen
    input_file = "./data_ensembl/3-187668812-187670494.txt"
    
    if not os.path.exists(input_file):
        print(f"ADVERTENCIA: Archivo de entrada {input_file} no encontrado.")
        # Verificar si los archivos de datos ya existen
        if not os.path.exists("data/data_ei.csv"):
            print("Creando datos de ejemplo para demostración...")
            # Aquí podrías crear datos sintéticos si lo necesitas
            # Por ahora, solo continuaremos con los siguientes pasos
    else:
        # Extraer datos genéticos
        data_paths = extract_genetic_data(input_file)
    
    # 2. Analizar cada tipo de transición genética
    transition_types = ['ei', 'ie', 'ze', 'ez']
    results = []
    
    for transition in transition_types:
        try:
            # Preparar datos para este tipo de transición
            true_data_path = f"data/data_{transition}.csv"
            false_data_path = f"data/data_{transition}_random.csv"
            
            if os.path.exists(true_data_path) and os.path.exists(false_data_path):
                X, y = prepare_transition_data(true_data_path, false_data_path)
                
                # Entrenar y evaluar modelos
                result = train_model_for_transition(transition, X, y)
                results.append(result)
            else:
                print(f"ADVERTENCIA: Archivos de datos para transición {transition} no encontrados.")
        except Exception as e:
            print(f"Error al procesar transición {transition}: {str(e)}")
    
    # 3. Crear leaderboard unificado
    if results:
        leaderboard = create_unified_leaderboard(results)
        print("\nLeaderboard unificado:")
        print(leaderboard)
        
        # 4. Crear visualizaciones
        create_visualizations(leaderboard, results)
    else:
        print("No se pudieron obtener resultados para ningún tipo de transición.")
    
    print("\nAnálisis de transiciones genómicas completado.")

if __name__ == "__main__":
    main() 