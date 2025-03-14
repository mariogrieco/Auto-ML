import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from predict import (
    predict_ei_transition, predict_ie_transition, 
    predict_ze_transition, predict_ez_transition,
    nucleotide_to_onehot
)

class GeneticTransitionEnsemble:
    """
    Ensemble para predicción de transiciones genómicas que combina
    los resultados de múltiples modelos especializados.
    """
    
    def __init__(self, models_dir='.'):
        """
        Inicializa el ensemble cargando los modelos disponibles.
        
        Args:
            models_dir: Directorio donde se encuentran los modelos guardados.
        """
        self.models = {}
        self.model_files = {
            'ei': 'best_ei_model.h5',
            'ie': 'best_ie_model.h5',
            'ze': 'best_ze_model.h5',
            'ez': 'best_ez_model.h5'
        }
        
        # Cargar los modelos disponibles
        for transition_type, model_file in self.model_files.items():
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    self.models[transition_type] = model_file
                    print(f"Modelo {transition_type} disponible: {model_file}")
                except Exception as e:
                    print(f"Error al cargar modelo {transition_type}: {e}")
            else:
                print(f"Modelo {transition_type} no encontrado en {model_path}")
    
    def get_available_models(self):
        """
        Retorna la lista de modelos disponibles.
        """
        return list(self.models.keys())
    
    def predict(self, sequence, transition_type=None):
        """
        Realiza una predicción utilizando el modelo adecuado según el tipo de transición.
        Si no se especifica un tipo, intenta determinar el tipo automáticamente.
        
        Args:
            sequence: Secuencia de nucleótidos.
            transition_type: Tipo de transición ('ei', 'ie', 'ze', 'ez', o None para auto).
            
        Returns:
            Diccionario con resultados de la predicción.
        """
        # Si no se especifica un tipo, determinarlo por la longitud
        if transition_type is None:
            seq_len = len(sequence)
            if seq_len == 12:
                transition_type = 'ei'
            elif seq_len == 105:
                transition_type = 'ie'
            elif seq_len == 550:
                # Ambiguo, podría ser ZE o EZ. Intentar ambos.
                results = []
                for t_type in ['ze', 'ez']:
                    if t_type in self.models:
                        try:
                            if t_type == 'ze':
                                result = predict_ze_transition(sequence)
                            else:
                                result = predict_ez_transition(sequence)
                            results.append((t_type, result))
                        except:
                            pass
                
                # Devolver el de mayor probabilidad
                if results:
                    results.sort(key=lambda x: x[1]['probability'], reverse=True)
                    return {
                        'transition_type': results[0][0],
                        'results': results[0][1],
                        'alternative_results': results[1:] if len(results) > 1 else []
                    }
                return {'error': 'No se pudo determinar el tipo de transición automáticamente'}
            else:
                return {'error': f'Longitud de secuencia {seq_len} no reconocida para ningún tipo de transición'}
        
        # Comprobar que el modelo solicitado esté disponible
        if transition_type not in self.models:
            return {'error': f'Modelo para transición {transition_type} no disponible'}
        
        # Realizar predicción según el tipo
        try:
            if transition_type == 'ei':
                result = predict_ei_transition(sequence)
            elif transition_type == 'ie':
                result = predict_ie_transition(sequence)
            elif transition_type == 'ze':
                result = predict_ze_transition(sequence)
            elif transition_type == 'ez':
                result = predict_ez_transition(sequence)
            else:
                return {'error': f'Tipo de transición {transition_type} no reconocido'}
            
            return {
                'transition_type': transition_type,
                'results': result
            }
        except Exception as e:
            return {'error': f'Error en predicción: {str(e)}'}
    
    def batch_predict(self, sequences_df, sequence_col, transition_type=None):
        """
        Realiza predicciones para un conjunto de secuencias.
        
        Args:
            sequences_df: DataFrame con las secuencias.
            sequence_col: Nombre de la columna que contiene las secuencias.
            transition_type: Tipo de transición (o None para auto).
            
        Returns:
            DataFrame con los resultados de las predicciones.
        """
        results = []
        
        for idx, row in sequences_df.iterrows():
            sequence = row[sequence_col]
            prediction = self.predict(sequence, transition_type)
            
            if 'error' in prediction:
                result_row = {
                    'sequence_id': idx,
                    'error': prediction['error']
                }
            else:
                pred_results = prediction['results']
                result_row = {
                    'sequence_id': idx,
                    'sequence': pred_results['sequence'],
                    'transition_type': prediction['transition_type'],
                    'probability': pred_results['probability'],
                    'prediction': pred_results['prediction'],
                    'confidence': pred_results['confidence']
                }
                
                # Añadir datos adicionales si existen en el DataFrame original
                for col in sequences_df.columns:
                    if col != sequence_col and col not in result_row:
                        result_row[col] = row[col]
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def evaluate_all_models(self, test_data_dir='./data'):
        """
        Evalúa todos los modelos disponibles en datos de prueba.
        
        Args:
            test_data_dir: Directorio donde se encuentran los datos de prueba.
            
        Returns:
            DataFrame con métricas de evaluación.
        """
        results = []
        
        for transition_type in self.models:
            # Cargar datos de prueba
            true_data_path = os.path.join(test_data_dir, f'data_{transition_type}.csv')
            false_data_path = os.path.join(test_data_dir, f'data_{transition_type}_random.csv')
            
            if not (os.path.exists(true_data_path) and os.path.exists(false_data_path)):
                print(f"Datos de prueba para {transition_type} no encontrados")
                continue
            
            # Cargar datos
            df_true = pd.read_csv(true_data_path)
            df_false = pd.read_csv(false_data_path)
            
            # Extraer columnas de nucleótidos
            feature_cols = [col for col in df_true.columns if col.startswith('B')]
            
            # Procesar ejemplos positivos
            true_count = 0
            true_total = len(df_true)
            true_probs = []
            
            for idx, row in df_true.iterrows():
                # Extraer secuencia
                sequence = ''.join([str(row[col]) for col in feature_cols])
                
                # Predecir
                prediction = self.predict(sequence, transition_type)
                if 'error' not in prediction:
                    prob = prediction['results']['probability']
                    true_probs.append(prob)
                    if prob > 0.5:
                        true_count += 1
            
            # Procesar ejemplos negativos
            false_count = 0
            false_total = len(df_false)
            false_probs = []
            
            for idx, row in df_false.iterrows():
                # Extraer secuencia
                sequence = ''.join([str(row[col]) for col in feature_cols])
                
                # Predecir
                prediction = self.predict(sequence, transition_type)
                if 'error' not in prediction:
                    prob = prediction['results']['probability']
                    false_probs.append(prob)
                    if prob <= 0.5:
                        false_count += 1
            
            # Calcular métricas
            accuracy = (true_count + false_count) / (true_total + false_total) if (true_total + false_total) > 0 else 0
            sensitivity = true_count / true_total if true_total > 0 else 0
            specificity = false_count / false_total if false_total > 0 else 0
            
            # Almacenar resultados
            results.append({
                'transition_type': transition_type,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'true_probs': true_probs,
                'false_probs': false_probs
            })
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame([{
            'transition_type': r['transition_type'],
            'accuracy': r['accuracy'],
            'sensitivity': r['sensitivity'],
            'specificity': r['specificity']
        } for r in results])
        
        # Graficar distribuciones de probabilidad
        self.plot_probability_distributions(results)
        
        return results_df
    
    def plot_probability_distributions(self, results):
        """
        Grafica las distribuciones de probabilidad para ejemplos positivos y negativos.
        
        Args:
            results: Lista de diccionarios con resultados de evaluación.
        """
        plt.figure(figsize=(15, 10))
        
        for i, result in enumerate(results):
            plt.subplot(2, 2, i+1)
            
            # Graficar histograma para ejemplos positivos
            if result['true_probs']:
                sns.histplot(result['true_probs'], bins=20, alpha=0.6, label='Ejemplos positivos', color='blue')
            
            # Graficar histograma para ejemplos negativos
            if result['false_probs']:
                sns.histplot(result['false_probs'], bins=20, alpha=0.6, label='Ejemplos negativos', color='red')
            
            plt.title(f"Distribución de probabilidades - {result['transition_type'].upper()}")
            plt.xlabel('Probabilidad')
            plt.ylabel('Frecuencia')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('probability_distributions.png')
        plt.show()
    
    def visualize_performance(self, eval_results=None, test_data_dir='./data'):
        """
        Visualiza el rendimiento de todos los modelos.
        
        Args:
            eval_results: DataFrame con resultados de evaluación previos.
            test_data_dir: Directorio con datos de prueba.
        """
        if eval_results is None:
            eval_results = self.evaluate_all_models(test_data_dir)
        
        # Gráfico de barras para métricas de rendimiento
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(eval_results))
        width = 0.25
        
        plt.bar(x - width, eval_results['accuracy'], width, label='Exactitud', color='blue')
        plt.bar(x, eval_results['sensitivity'], width, label='Sensibilidad', color='green')
        plt.bar(x + width, eval_results['specificity'], width, label='Especificidad', color='orange')
        
        plt.xlabel('Tipo de transición')
        plt.ylabel('Puntuación')
        plt.title('Rendimiento de los modelos por tipo de transición')
        plt.xticks(x, eval_results['transition_type'].str.upper())
        plt.legend()
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.show()
        
        # Heatmap para la matriz de rendimiento
        plt.figure(figsize=(10, 6))
        
        # Preparar datos para el heatmap
        heatmap_data = eval_results.pivot(index='transition_type', columns=None, values=['accuracy', 'sensitivity', 'specificity'])
        
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
        plt.title('Matriz de rendimiento de los modelos')
        plt.tight_layout()
        plt.savefig('performance_heatmap.png')
        plt.show()


if __name__ == "__main__":
    print("Ensemble de modelos para detección de transiciones genómicas")
    print("=" * 60)
    
    # Inicializar el ensemble
    ensemble = GeneticTransitionEnsemble()
    
    # Mostrar modelos disponibles
    available_models = ensemble.get_available_models()
    print(f"Modelos disponibles: {', '.join(available_models)}")
    
    # Si hay al menos un modelo disponible, podemos proceder
    if available_models:
        print("\nEvaluando rendimiento de los modelos...")
        eval_results = ensemble.evaluate_all_models()
        
        print("\nResumen de rendimiento:")
        print(eval_results)
        
        print("\nVisualizando rendimiento...")
        ensemble.visualize_performance(eval_results)
        
        # Ejemplos de predicción individual
        print("\nEjemplos de predicción:")
        
        # EI
        if 'ei' in available_models:
            ei_sequence = "aagctGTaagct"
            print(f"\nSecuencia EI: {ei_sequence}")
            ei_result = ensemble.predict(ei_sequence)
            print(f"Predicción: {ei_result['results']['prediction']}")
            print(f"Probabilidad: {ei_result['results']['probability']:.4f}")
        
        # IE
        if 'ie' in available_models:
            ie_sequence = "c" * 100 + "agAAA"
            print(f"\nSecuencia IE: {ie_sequence[:10]}...{ie_sequence[-5:]}")
            ie_result = ensemble.predict(ie_sequence)
            print(f"Predicción: {ie_result['results']['prediction']}")
            print(f"Probabilidad: {ie_result['results']['probability']:.4f}")
        
        # Ejemplo de detección automática del tipo de transición
        if 'ze' in available_models and 'ez' in available_models:
            print("\nDetección automática del tipo de transición:")
            auto_sequence = "a" * 500 + "t" * 50
            print(f"Secuencia (550nt): {auto_sequence[:5]}...{auto_sequence[-5:]}")
            auto_result = ensemble.predict(auto_sequence)
            
            print(f"Tipo detectado: {auto_result['transition_type']}")
            print(f"Predicción: {auto_result['results']['prediction']}")
            print(f"Probabilidad: {auto_result['results']['probability']:.4f}")
            
            if 'alternative_results' in auto_result and auto_result['alternative_results']:
                alt = auto_result['alternative_results'][0]
                print(f"Alternativa: {alt[0]}, Probabilidad: {alt[1]['probability']:.4f}")
    else:
        print("No hay modelos disponibles. Entrena los modelos primero con train_models.py.") 