import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Definición de funciones de utilidad

def load_and_prepare_data(true_data_path, false_data_path, test_size=0.2, val_size=0.1):
    """
    Carga los datos de los archivos CSV, los combina y los divide en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        true_data_path: Ruta al archivo CSV con los ejemplos positivos.
        false_data_path: Ruta al archivo CSV con los ejemplos negativos.
        test_size: Proporción de datos para pruebas.
        val_size: Proporción de datos para validación.
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Conjuntos de datos divididos.
    """
    # Cargar datos
    df_true = pd.read_csv(true_data_path)
    df_false = pd.read_csv(false_data_path)
    
    # Extraer las columnas que contienen nucleótidos (columnas 'B1', 'B2', ...)
    feature_cols = [col for col in df_true.columns if col.startswith('B')]
    
    # Combinar datos y crear etiquetas
    X_true = df_true[feature_cols].values
    X_false = df_false[feature_cols].values
    
    y_true = np.ones(len(X_true))
    y_false = np.zeros(len(X_false))
    
    # Combinar los datos
    X = np.vstack((X_true, X_false))
    y = np.concatenate((y_true, y_false))
    
    # Preprocesar: convertir caracteres de nucleótidos a números
    # A->1, C->2, G->3, T/U->4, Otros->0
    nucleotide_map = {'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4}
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if isinstance(X[i, j], str):
                X[i, j] = nucleotide_map.get(X[i, j].lower(), 0)
    
    # Hacer one-hot encoding
    X_onehot = tf.one_hot(X.astype(np.int32), depth=5)  # 5 para 0,A,C,G,T
    
    # Dividir en train, val y test
    X_temp, X_test, y_temp, y_test = train_test_split(X_onehot, y, test_size=test_size, random_state=42)
    
    # Calcular el tamaño del conjunto de validación respecto al tamaño original
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_training_history(history):
    """
    Grafica la precisión y la pérdida durante el entrenamiento.
    
    Args:
        history: Objeto History devuelto por model.fit().
    """
    plt.figure(figsize=(12, 5))
    
    # Graficar precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
    
    # Graficar pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y muestra métricas de rendimiento.
    
    Args:
        model: Modelo entrenado.
        X_test: Datos de prueba.
        y_test: Etiquetas de prueba.
        
    Returns:
        Diccionario con las métricas de evaluación.
    """
    # Predecir probabilidades
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)
    
    # Informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Área ROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    return {
        'accuracy': (cm[0, 0] + cm[1, 1]) / np.sum(cm),
        'sensitivity': cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0,
        'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
        'auc': roc_auc
    }


# Definición de modelos

def create_cnn_model(input_shape):
    """
    Crea un modelo CNN para clasificación de secuencias.
    
    Args:
        input_shape: Forma de los datos de entrada.
        
    Returns:
        Modelo compilado.
    """
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.25),
        
        Conv1D(128, 3, activation='relu'),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(input_shape):
    """
    Crea un modelo LSTM bidireccional para clasificación de secuencias.
    
    Args:
        input_shape: Forma de los datos de entrada.
        
    Returns:
        Modelo compilado.
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.25),
        Bidirectional(LSTM(64)),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_hybrid_model(input_shape):
    """
    Crea un modelo híbrido CNN+LSTM para clasificación de secuencias.
    
    Args:
        input_shape: Forma de los datos de entrada.
        
    Returns:
        Modelo compilado.
    """
    inputs = Input(shape=input_shape)
    
    # Rama CNN
    x1 = Conv1D(64, 3, activation='relu')(inputs)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(128, 3, activation='relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    
    # Rama LSTM
    x2 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x2 = Bidirectional(LSTM(64))(x2)
    
    # Combinar ambas ramas
    x = tf.keras.layers.concatenate([Flatten()(x1), x2])
    
    # Capas densas finales
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Implementación para el modelo EI (Exón → Intrón)

def train_ei_model():
    """
    Entrena y evalúa un modelo para la transición Exón → Intrón.
    """
    print("Entrenando modelo para transición Exón → Intrón (EI)")
    
    # Cargar datos
    data_path_true = "./data/data_ei.csv"
    data_path_false = "./data/data_ei_random.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        data_path_true, data_path_false
    )
    
    # Obtener la forma de los datos
    input_shape = X_train.shape[1:]
    print(f"Forma de los datos de entrada: {input_shape}")
    
    # Crear modelo (puede elegir entre CNN, LSTM o Híbrido)
    model = create_cnn_model(input_shape)
    #model = create_lstm_model(input_shape)
    #model = create_hybrid_model(input_shape)
    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_ei_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Visualizar historia de entrenamiento
    plot_training_history(history)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Exactitud: {metrics['accuracy']:.4f}")
    print(f"Sensibilidad: {metrics['sensitivity']:.4f}")
    print(f"Especificidad: {metrics['specificity']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return model, metrics


# Implementación para el modelo IE (Intrón → Exón)

def train_ie_model():
    """
    Entrena y evalúa un modelo para la transición Intrón → Exón.
    """
    print("Entrenando modelo para transición Intrón → Exón (IE)")
    
    # Cargar datos
    data_path_true = "./data/data_ie.csv"
    data_path_false = "./data/data_ie_random.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        data_path_true, data_path_false
    )
    
    # Obtener la forma de los datos
    input_shape = X_train.shape[1:]
    print(f"Forma de los datos de entrada: {input_shape}")
    
    # Crear modelo (puede elegir entre CNN, LSTM o Híbrido)
    #model = create_cnn_model(input_shape)
    model = create_lstm_model(input_shape)
    #model = create_hybrid_model(input_shape)
    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_ie_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Visualizar historia de entrenamiento
    plot_training_history(history)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Exactitud: {metrics['accuracy']:.4f}")
    print(f"Sensibilidad: {metrics['sensitivity']:.4f}")
    print(f"Especificidad: {metrics['specificity']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return model, metrics


# Implementación para el modelo ZE (Zona Intergénica → Primer Exón)

def train_ze_model():
    """
    Entrena y evalúa un modelo para la transición Zona Intergénica → Primer Exón.
    """
    print("Entrenando modelo para transición Zona Intergénica → Primer Exón (ZE)")
    
    # Cargar datos
    data_path_true = "./data/data_ze.csv"
    data_path_false = "./data/data_ze_random.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        data_path_true, data_path_false
    )
    
    # Obtener la forma de los datos
    input_shape = X_train.shape[1:]
    print(f"Forma de los datos de entrada: {input_shape}")
    
    # Para secuencias largas (550 nucleótidos), un modelo híbrido sería más adecuado
    model = create_hybrid_model(input_shape)
    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_ze_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,  # Tamaño de batch más pequeño para secuencias largas
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Visualizar historia de entrenamiento
    plot_training_history(history)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Exactitud: {metrics['accuracy']:.4f}")
    print(f"Sensibilidad: {metrics['sensitivity']:.4f}")
    print(f"Especificidad: {metrics['specificity']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return model, metrics


# Implementación para el modelo EZ (Último Exón → Zona Intergénica)

def train_ez_model():
    """
    Entrena y evalúa un modelo para la transición Último Exón → Zona Intergénica.
    """
    print("Entrenando modelo para transición Último Exón → Zona Intergénica (EZ)")
    
    # Cargar datos
    data_path_true = "./data/data_ez.csv"
    data_path_false = "./data/data_ez_random.csv"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        data_path_true, data_path_false
    )
    
    # Obtener la forma de los datos
    input_shape = X_train.shape[1:]
    print(f"Forma de los datos de entrada: {input_shape}")
    
    # Para secuencias largas (550 nucleótidos), un modelo híbrido sería más adecuado
    model = create_hybrid_model(input_shape)
    
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_ez_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,  # Tamaño de batch más pequeño para secuencias largas
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Visualizar historia de entrenamiento
    plot_training_history(history)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Exactitud: {metrics['accuracy']:.4f}")
    print(f"Sensibilidad: {metrics['sensitivity']:.4f}")
    print(f"Especificidad: {metrics['specificity']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    print("AutoML para detección de zonas de transición genómica")
    print("=" * 50)
    
    # Entrenar modelo EI
    ei_model, ei_metrics = train_ei_model()
    
    print("\n" + "=" * 50 + "\n")
    
    # Entrenar modelo IE
    ie_model, ie_metrics = train_ie_model()
    
    print("\n" + "=" * 50 + "\n")
    
    # Entrenar modelo ZE
    ze_model, ze_metrics = train_ze_model()
    
    print("\n" + "=" * 50 + "\n")
    
    # Entrenar modelo EZ
    ez_model, ez_metrics = train_ez_model()
    
    # Comparar resultados
    print("\nComparación de modelos:")
    print("-" * 50)
    print(f"Modelo EI - AUC: {ei_metrics['auc']:.4f}, Exactitud: {ei_metrics['accuracy']:.4f}")
    print(f"Modelo IE - AUC: {ie_metrics['auc']:.4f}, Exactitud: {ie_metrics['accuracy']:.4f}")
    print(f"Modelo ZE - AUC: {ze_metrics['auc']:.4f}, Exactitud: {ze_metrics['accuracy']:.4f}")
    print(f"Modelo EZ - AUC: {ez_metrics['auc']:.4f}, Exactitud: {ez_metrics['accuracy']:.4f}") 