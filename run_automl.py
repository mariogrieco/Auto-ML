#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AutoML para transiciones genómicas - Script principal
Este script orquesta todo el flujo de trabajo, desde la extracción de datos
hasta el entrenamiento de modelos y análisis de resultados.
"""

import os
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_arguments():
    """
    Analiza los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description='AutoML para transiciones genómicas')
    
    parser.add_argument('--extract', action='store_true',
                        help='Ejecutar extracción de datos')
    
    parser.add_argument('--train', action='store_true',
                        help='Entrenar modelos')
    
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluar modelos')
    
    parser.add_argument('--all', action='store_true',
                        help='Ejecutar todo el flujo de trabajo')
    
    parser.add_argument('--input_file', type=str, default='./data_ensembl/3-187668812-187670494.txt',
                        help='Archivo de entrada para extracción (default: %(default)s)')
    
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directorio para guardar resultados (default: %(default)s)')
    
    return parser.parse_args()

def run_extraction(input_file, output_dir):
    """
    Ejecuta el proceso de extracción de datos.
    """
    print(f"\n{'='*20} EXTRACCIÓN DE DATOS {'='*20}\n")
    
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Ejecutar el extractor
    print(f"Ejecutando extracción desde {input_file}...")
    try:
        from main_extractor import Extraction
        extractor = Extraction(input_file, output_dir)
        extractor.process_file()
        extractor.save_to_csv()
        print("Extracción completada con éxito.")
        
        # Verificar archivos generados
        check_extracted_files(output_dir)
    except Exception as e:
        print(f"Error durante la extracción: {e}")
        return False
    
    return True

def check_extracted_files(output_dir):
    """
    Verifica que los archivos de datos extraídos existen y muestra información.
    """
    file_patterns = [
        'data_ei.csv', 'data_ei_random.csv',
        'data_ie.csv', 'data_ie_random.csv',
        'data_ze.csv', 'data_ze_random.csv',
        'data_ez.csv', 'data_ez_random.csv'
    ]
    
    all_exist = True
    file_counts = {}
    
    for pattern in file_patterns:
        file_path = os.path.join(output_dir, pattern)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                count = len(df)
                file_counts[pattern] = count
                print(f" - {pattern}: {count} registros")
            except:
                print(f" - {pattern}: Error al leer archivo")
                all_exist = False
        else:
            print(f" - {pattern}: No encontrado")
            all_exist = False
    
    if all_exist:
        # Visualizar distribución de datos extraídos
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            [p.replace('.csv', '').replace('data_', '') for p in file_patterns],
            [file_counts.get(p, 0) for p in file_patterns]
        )
        
        # Añadir etiquetas
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height}',
                ha='center', va='bottom'
            )
        
        plt.title('Distribución de datos extraídos')
        plt.xlabel('Tipo de transición')
        plt.ylabel('Número de secuencias')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
        plt.close()
    
    return all_exist

def run_training():
    """
    Ejecuta el entrenamiento de modelos.
    """
    print(f"\n{'='*20} ENTRENAMIENTO DE MODELOS {'='*20}\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"train_log_{timestamp}.txt"
    
    print(f"Iniciando entrenamiento (log: {log_file})...")
    
    try:
        # Ejecutar el script de entrenamiento
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ['python', 'train_models.py'],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait()
        
        if process.returncode == 0:
            print("Entrenamiento completado con éxito.")
        else:
            print(f"Error durante el entrenamiento. Código de salida: {process.returncode}")
            return False
        
        # Verificar modelos generados
        check_models()
    except Exception as e:
        print(f"Error al iniciar el entrenamiento: {e}")
        return False
    
    return True

def check_models():
    """
    Verifica que los modelos entrenados existen.
    """
    model_files = [
        'best_ei_model.h5',
        'best_ie_model.h5',
        'best_ze_model.h5',
        'best_ez_model.h5'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f" - {model_file}: {size_mb:.2f} MB")
        else:
            print(f" - {model_file}: No encontrado")

def run_evaluation():
    """
    Ejecuta la evaluación de modelos.
    """
    print(f"\n{'='*20} EVALUACIÓN DE MODELOS {'='*20}\n")
    
    try:
        # Ejecutar el script de evaluación
        subprocess.run(['python', 'ensemble_predict.py'], check=True)
        print("Evaluación completada con éxito.")
    except subprocess.CalledProcessError as e:
        print(f"Error durante la evaluación. Código de salida: {e.returncode}")
        return False
    except Exception as e:
        print(f"Error al iniciar la evaluación: {e}")
        return False
    
    return True

def main():
    """
    Función principal que orquesta todo el flujo de trabajo.
    """
    args = parse_arguments()
    
    # Si no se especificó ninguna operación, mostrar ayuda
    if not (args.extract or args.train or args.evaluate or args.all):
        print("No se especificó ninguna operación. Use --help para ver las opciones disponibles.")
        return
    
    # Registrar inicio
    print(f"\nAutoML para transiciones genómicas - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Ejecutar operaciones según argumentos
    if args.all or args.extract:
        extraction_success = run_extraction(args.input_file, args.output_dir)
        if not extraction_success and args.all:
            print("\nDetención temprana debido a errores en la extracción.")
            return
    
    if args.all or args.train:
        training_success = run_training()
        if not training_success and args.all:
            print("\nDetención temprana debido a errores en el entrenamiento.")
            return
    
    if args.all or args.evaluate:
        evaluation_success = run_evaluation()
    
    print(f"\n{'='*60}")
    print(f"Proceso completado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 