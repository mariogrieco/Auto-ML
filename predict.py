import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

def nucleotide_to_onehot(sequence):
    """
    Convierte una secuencia de nucleótidos en representación one-hot.
    
    Args:
        sequence: Lista o array de nucleótidos (caracteres 'A', 'C', 'G', 'T').
        
    Returns:
        Array numpy con codificación one-hot.
    """
    # Convertir a números primero
    nucleotide_map = {'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4}
    
    # Inicializar array numérico
    numeric_sequence = np.zeros(len(sequence))
    
    # Convertir cada nucleótido
    for i, nucleotide in enumerate(sequence):
        if isinstance(nucleotide, str):
            numeric_sequence[i] = nucleotide_map.get(nucleotide.lower(), 0)
    
    # Convertir a one-hot
    return tf.one_hot(numeric_sequence.astype(np.int32), depth=5)

def predict_ei_transition(sequence, model_path='best_ei_model.h5'):
    """
    Predice si una secuencia representa una transición Exón → Intrón.
    
    Args:
        sequence: Lista o cadena de 12 nucleótidos.
        model_path: Ruta al modelo guardado.
        
    Returns:
        Probabilidad de que la secuencia sea una transición EI válida.
    """
    # Cargar modelo
    model = load_model(model_path)
    
    # Procesar secuencia
    if isinstance(sequence, str):
        sequence = list(sequence.lower())
    
    # Comprobar longitud
    if len(sequence) != 12:
        raise ValueError("La secuencia para EI debe tener 12 nucleótidos (5 izquierda + 7 derecha)")
    
    # Convertir a one-hot
    x = nucleotide_to_onehot(sequence)
    
    # Agregar dimensión de batch
    x = tf.expand_dims(x, axis=0)
    
    # Hacer predicción
    prediction = model.predict(x)[0][0]
    
    return {
        'sequence': ''.join(sequence),
        'probability': float(prediction),
        'prediction': 'Transición EI válida' if prediction > 0.5 else 'No es una transición EI válida',
        'confidence': max(prediction, 1-prediction)
    }

def predict_ie_transition(sequence, model_path='best_ie_model.h5'):
    """
    Predice si una secuencia representa una transición Intrón → Exón.
    
    Args:
        sequence: Lista o cadena de 105 nucleótidos.
        model_path: Ruta al modelo guardado.
        
    Returns:
        Probabilidad de que la secuencia sea una transición IE válida.
    """
    # Cargar modelo
    model = load_model(model_path)
    
    # Procesar secuencia
    if isinstance(sequence, str):
        sequence = list(sequence.lower())
    
    # Comprobar longitud
    if len(sequence) != 105:
        raise ValueError("La secuencia para IE debe tener 105 nucleótidos (100 izquierda + 5 derecha)")
    
    # Convertir a one-hot
    x = nucleotide_to_onehot(sequence)
    
    # Agregar dimensión de batch
    x = tf.expand_dims(x, axis=0)
    
    # Hacer predicción
    prediction = model.predict(x)[0][0]
    
    return {
        'sequence': ''.join(sequence),
        'probability': float(prediction),
        'prediction': 'Transición IE válida' if prediction > 0.5 else 'No es una transición IE válida',
        'confidence': max(prediction, 1-prediction)
    }

def predict_ze_transition(sequence, model_path='best_ze_model.h5'):
    """
    Predice si una secuencia representa una transición Zona Intergénica → Primer Exón.
    
    Args:
        sequence: Lista o cadena de 550 nucleótidos.
        model_path: Ruta al modelo guardado.
        
    Returns:
        Probabilidad de que la secuencia sea una transición ZE válida.
    """
    # Cargar modelo
    model = load_model(model_path)
    
    # Procesar secuencia
    if isinstance(sequence, str):
        sequence = list(sequence.lower())
    
    # Comprobar longitud
    if len(sequence) != 550:
        raise ValueError("La secuencia para ZE debe tener 550 nucleótidos (500 izquierda + 50 derecha)")
    
    # Convertir a one-hot
    x = nucleotide_to_onehot(sequence)
    
    # Agregar dimensión de batch
    x = tf.expand_dims(x, axis=0)
    
    # Hacer predicción
    prediction = model.predict(x)[0][0]
    
    return {
        'sequence': ''.join(sequence),
        'probability': float(prediction),
        'prediction': 'Transición ZE válida' if prediction > 0.5 else 'No es una transición ZE válida',
        'confidence': max(prediction, 1-prediction)
    }

def predict_ez_transition(sequence, model_path='best_ez_model.h5'):
    """
    Predice si una secuencia representa una transición Último Exón → Zona Intergénica.
    
    Args:
        sequence: Lista o cadena de 550 nucleótidos.
        model_path: Ruta al modelo guardado.
        
    Returns:
        Probabilidad de que la secuencia sea una transición EZ válida.
    """
    # Cargar modelo
    model = load_model(model_path)
    
    # Procesar secuencia
    if isinstance(sequence, str):
        sequence = list(sequence.lower())
    
    # Comprobar longitud
    if len(sequence) != 550:
        raise ValueError("La secuencia para EZ debe tener 550 nucleótidos (50 izquierda + 500 derecha)")
    
    # Convertir a one-hot
    x = nucleotide_to_onehot(sequence)
    
    # Agregar dimensión de batch
    x = tf.expand_dims(x, axis=0)
    
    # Hacer predicción
    prediction = model.predict(x)[0][0]
    
    return {
        'sequence': ''.join(sequence),
        'probability': float(prediction),
        'prediction': 'Transición EZ válida' if prediction > 0.5 else 'No es una transición EZ válida',
        'confidence': max(prediction, 1-prediction)
    }

def process_fasta_file(fasta_path, output_path, transition_type='ei'):
    """
    Procesa un archivo FASTA y realiza predicciones para cada secuencia.
    
    Args:
        fasta_path: Ruta al archivo FASTA.
        output_path: Ruta donde guardar los resultados.
        transition_type: Tipo de transición a predecir ('ei', 'ie', 'ze' o 'ez').
    """
    # Diccionario para almacenar las secuencias
    sequences = {}
    current_id = None
    
    # Leer archivo FASTA
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
                sequences[current_id] = ''
            else:
                sequences[current_id] += line
    
    # Resultados
    results = []
    
    # Función de predicción según el tipo
    if transition_type.lower() == 'ei':
        predict_func = predict_ei_transition
        expected_length = 12
    elif transition_type.lower() == 'ie':
        predict_func = predict_ie_transition
        expected_length = 105
    elif transition_type.lower() == 'ze':
        predict_func = predict_ze_transition
        expected_length = 550
    elif transition_type.lower() == 'ez':
        predict_func = predict_ez_transition
        expected_length = 550
    else:
        raise ValueError("Tipo de transición no reconocido. Debe ser 'ei', 'ie', 'ze' o 'ez'.")
    
    # Realizar predicciones
    for seq_id, sequence in sequences.items():
        if len(sequence) != expected_length:
            print(f"Advertencia: La secuencia {seq_id} tiene longitud {len(sequence)}, pero se esperaba {expected_length}.")
            continue
        
        prediction = predict_func(sequence)
        results.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'probability': prediction['probability'],
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence']
        })
    
    # Guardar resultados
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Resultados guardados en {output_path}")

if __name__ == "__main__":
    # Ejemplo de uso
    print("Predicción de transiciones genómicas")
    print("=" * 50)
    
    # Ejemplo de predicción EI
    ei_sequence = "aagctGTaagct"  # 5 nucleótidos a la izquierda + GT + 5 a la derecha
    ei_result = predict_ei_transition(ei_sequence)
    print(f"Secuencia EI: {ei_sequence}")
    print(f"Probabilidad: {ei_result['probability']:.4f}")
    print(f"Predicción: {ei_result['prediction']}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Ejemplo de predicción IE
    # En un ejemplo real, sería una secuencia de 105 nucleótidos
    ie_sequence = "c" * 100 + "agAAA"  # 100 nucleótidos + AG + 3 a la derecha
    ie_result = predict_ie_transition(ie_sequence)
    print(f"Secuencia IE: {ie_sequence[:10]}...{ie_sequence[-5:]}")
    print(f"Probabilidad: {ie_result['probability']:.4f}")
    print(f"Predicción: {ie_result['prediction']}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Ejemplo de predicción ZE
    # En un ejemplo real, sería una secuencia de 550 nucleótidos
    ze_sequence = "a" * 500 + "c" * 50  # 500 nucleótidos intergénicos + 50 del primer exón
    try:
        ze_result = predict_ze_transition(ze_sequence)
        print(f"Secuencia ZE: {ze_sequence[:10]}...{ze_sequence[-10:]}")
        print(f"Probabilidad: {ze_result['probability']:.4f}")
        print(f"Predicción: {ze_result['prediction']}")
    except Exception as e:
        print(f"Error en predicción ZE: {e}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Ejemplo de predicción EZ
    # En un ejemplo real, sería una secuencia de 550 nucleótidos
    ez_sequence = "g" * 50 + "t" * 500  # 50 nucleótidos del último exón + 500 intergénicos
    try:
        ez_result = predict_ez_transition(ez_sequence)
        print(f"Secuencia EZ: {ez_sequence[:10]}...{ez_sequence[-10:]}")
        print(f"Probabilidad: {ez_result['probability']:.4f}")
        print(f"Predicción: {ez_result['prediction']}")
    except Exception as e:
        print(f"Error en predicción EZ: {e}")
    
    # Ejemplo de procesamiento de archivo FASTA
    # Descomenta estas líneas para usar con un archivo FASTA real
    # process_fasta_file(
    #     "secuencias_ei.fasta", 
    #     "resultados_ei.csv", 
    #     transition_type='ei'
    # ) 