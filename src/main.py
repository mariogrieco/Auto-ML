import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuración de MLflow
mlflow.set_tracking_uri("./mlflow")
mlflow.set_experiment("automl-genetic-prediction")

def load_data(filepath):
    """Carga los datos de entrada"""
    print(f"Cargando datos desde {filepath}")
    # Aquí iría la lógica para cargar datos genéticos
    # Ejemplo simplificado:
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocesa los datos para el modelado"""
    print("Preprocesando datos...")
    # Aquí iría la lógica de preprocesamiento específica para datos genéticos
    # Ejemplo simplificado:
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def train_automl_models(X_train, y_train, X_test, y_test):
    """Entrena múltiples modelos AutoML y los compara"""
    print("Entrenando modelos AutoML...")
    
    # Diccionario para almacenar resultados de modelos
    model_results = {}
    
    # 1. TPOT
    try:
        print("\nEntrenando con TPOT...")
        from tpot import TPOTClassifier
        
        with mlflow.start_run(run_name="tpot-automl"):
            tpot = TPOTClassifier(generations=5, 
                                  population_size=20,
                                  verbosity=2,
                                  random_state=42,
                                  config_dict='TPOT sparse')
            
            tpot.fit(X_train, y_train)
            y_pred = tpot.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Guardar resultados
            model_results['TPOT'] = {
                'model': tpot,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            }
            
            # Log del modelo
            mlflow.sklearn.log_model(tpot.fitted_pipeline_, "tpot_model")
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    except Exception as e:
        print(f"Error con TPOT: {e}")
    
    # 2. LazyPredict (reemplazo para Auto-sklearn que no es compatible con Windows)
    try:
        print("\nEntrenando con LazyPredict...")
        from lazypredict.Supervised import LazyClassifier
        
        with mlflow.start_run(run_name="lazypredict-automl"):
            lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True)
            models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
            
            # Obtener el mejor modelo
            best_model_name = models.index[0]
            best_model_accuracy = models.iloc[0]['Accuracy']
            
            # Obtener otras métricas si están disponibles
            # LazyPredict puede no tener todas las métricas
            metrics_dict = {}
            metrics_dict['accuracy'] = best_model_accuracy
            
            for metric in ['Balanced Accuracy', 'F1 Score', 'Precision', 'Recall']:
                if metric in models.columns:
                    metric_key = metric.lower().replace(' ', '_')
                    metrics_dict[metric_key] = models.iloc[0][metric]
            
            metrics_dict['name'] = best_model_name
            
            # Guardar resultados
            model_results['LazyPredict'] = {
                'model': None,  # No podemos acceder directamente al modelo
                'metrics': metrics_dict
            }
            
            # Log de las métricas
            mlflow_metrics = {k: v for k, v in metrics_dict.items() if k != 'name' and v is not None}
            mlflow.log_metrics(mlflow_metrics)
            mlflow.log_param('best_model_name', best_model_name)
            
            # Guardar el DataFrame completo de resultados como CSV
            models.to_csv('lazypredict_models.csv')
            mlflow.log_artifact('lazypredict_models.csv')
            
    except Exception as e:
        print(f"Error con LazyPredict: {e}")
    
    # 3. FLAML
    try:
        print("\nEntrenando con FLAML...")
        from flaml import AutoML
        
        with mlflow.start_run(run_name="flaml-automl"):
            automl = AutoML()
            automl.fit(X_train, y_train, 
                       task="classification",
                       metric="accuracy", 
                       time_budget=300,  # 5 minutos
                       verbose=1)
            
            y_pred = automl.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Guardar resultados
            model_results['FLAML'] = {
                'model': automl,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            }
            
            # Log del modelo
            mlflow.sklearn.log_model(automl, "flaml_model")
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            mlflow.log_params(automl.best_config)
            mlflow.log_param('best_estimator', automl.best_estimator)
    except Exception as e:
        print(f"Error con FLAML: {e}")
    
    return model_results

def create_leaderboard(model_results):
    """Crea un leaderboard comparativo con todos los modelos"""
    print("Creando leaderboard...")
    
    leaderboard_data = []
    
    for framework, result in model_results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            
            # Extraer métricas según el formato en que se hayan guardado
            if isinstance(metrics, dict):
                entry = {
                    'Framework': framework,
                    'Accuracy': metrics.get('accuracy', None),
                    'Precision': metrics.get('precision', None),
                    'Recall': metrics.get('recall', None),
                    'F1': metrics.get('f1', None),
                    'AUC': metrics.get('auc', None)
                }
                # Si es LazyPredict, guardar también el nombre del mejor modelo
                if framework == 'LazyPredict' and 'name' in metrics:
                    entry['Best Model'] = metrics['name']
            else:  # DataFrame (como en PyCaret)
                entry = {
                    'Framework': framework,
                    'Accuracy': metrics.loc[0, 'Accuracy'] if 'Accuracy' in metrics.columns else None,
                    'Precision': metrics.loc[0, 'Prec.'] if 'Prec.' in metrics.columns else None,
                    'Recall': metrics.loc[0, 'Recall'] if 'Recall' in metrics.columns else None,
                    'F1': metrics.loc[0, 'F1'] if 'F1' in metrics.columns else None,
                    'AUC': metrics.loc[0, 'AUC'] if 'AUC' in metrics.columns else None
                }
            
            leaderboard_data.append(entry)
    
    # Crear dataframe del leaderboard
    leaderboard = pd.DataFrame(leaderboard_data)
    
    # Ordenar por accuracy (o la métrica preferida)
    if 'Accuracy' in leaderboard.columns and not leaderboard.empty:
        leaderboard = leaderboard.sort_values('Accuracy', ascending=False)
    
    # Guardar leaderboard
    leaderboard.to_csv("leaderboard.csv", index=False)
    print(f"Leaderboard guardado en leaderboard.csv")
    
    # Si todos los resultados de LazyPredict están disponibles, también guardamos ese leaderboard detallado
    if os.path.exists('lazypredict_models.csv'):
        print(f"Leaderboard detallado de LazyPredict guardado en lazypredict_models.csv")
    
    return leaderboard

def deploy_best_model(model_results, leaderboard):
    """Despliega el mejor modelo usando MLflow"""
    print("Desplegando el mejor modelo...")
    
    if leaderboard.empty:
        print("No hay modelos para desplegar. El leaderboard está vacío.")
        return
    
    # Identificar el mejor modelo según el leaderboard
    best_framework = leaderboard.iloc[0]['Framework']
    print(f"El mejor modelo es: {best_framework}")
    
    # Si el mejor modelo es LazyPredict, no podemos desplegarlo directamente
    if best_framework == 'LazyPredict':
        print("ADVERTENCIA: LazyPredict no proporciona acceso al modelo entrenado para despliegue.")
        print("Se recomienda entrenar manualmente el modelo identificado como mejor en LazyPredict.")
        if 'Best Model' in leaderboard.columns:
            print(f"El mejor modelo en LazyPredict fue: {leaderboard.iloc[0]['Best Model']}")
        return
    
    best_model = model_results[best_framework]['model']
    
    # Registrar el modelo en MLflow
    with mlflow.start_run(run_name=f"deploy-{best_framework}"):
        # Para todos los frameworks
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="genetic_structure_prediction"
        )
    
    print(f"Modelo {best_framework} desplegado y registrado como 'genetic_structure_prediction'")

def main():
    # Configurar directorios
    if not os.path.exists("mlflow"):
        os.makedirs("mlflow")
    
    # 1. Cargar datos
    # Nota: Para ejecutar esto, necesitarás un dataset real
    # Aquí deberías reemplazar con la ruta a tus datos genéticos
    try:
        data = load_data("data/processed/genetic_data.csv")
    except FileNotFoundError:
        print("ADVERTENCIA: Archivo de datos no encontrado. Creando datos de ejemplo para demostración...")
        # Crear datos de ejemplo para demostración
        np.random.seed(42)
        X = np.random.rand(1000, 20)  # 1000 muestras, 20 características
        y = np.random.randint(0, 2, 1000)  # Clasificación binaria
        
        # Convertir a DataFrame
        cols = [f'feature_{i}' for i in range(20)]
        data = pd.DataFrame(X, columns=cols)
        data['target'] = y
        
        # Guardar datos de ejemplo
        os.makedirs("data/processed", exist_ok=True)
        data.to_csv("data/processed/genetic_data.csv", index=False)
        print("Datos de ejemplo creados y guardados.")
    
    # 2. Preprocesar datos
    X, y = preprocess_data(data)
    
    # 3. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenar modelos AutoML
    model_results = train_automl_models(X_train, y_train, X_test, y_test)
    
    # 5. Crear leaderboard
    leaderboard = create_leaderboard(model_results)
    print("\nLeaderboard de modelos:")
    print(leaderboard)
    
    # 6. Desplegar el mejor modelo
    deploy_best_model(model_results, leaderboard)
    
    print("\nProceso completado con éxito!")

if __name__ == "__main__":
    main() 