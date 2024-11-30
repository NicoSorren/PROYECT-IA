from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
import os
import librosa
import numpy as np

def main():
    # Paso 1: Procesar el archivo de audio de prueba 'papa_prueba.ogg' utilizando AudioProcessor
    archivo_audio = "papa_prueba.ogg"
    procesador = AudioProcessor(input_folder="TempAudios", output_folder="TempAudios")
    procesador.eliminar_silencios(archivo_audio)  # Procesar el archivo y almacenarlo en TempAudios como .wav

    # Verificar si el archivo procesado está disponible
    archivo_procesado = "TempAudios/procesado_papa_prueba.wav"
    if os.path.exists(archivo_procesado):
        print(f"El archivo procesado se ha guardado correctamente como {archivo_procesado}")
    else:
        print(f"Error: El archivo procesado {archivo_procesado} no se ha guardado correctamente.")
        return

    # Paso 2: Extraer características del archivo procesado utilizando FeatureExtractor
    extractor = FeatureExtractor(input_folder="TempAudios", use_pca=True, n_components=9)
    
    try:
        # Cargar el audio desde el archivo procesado
        audio, sample_rate = librosa.load(archivo_procesado, sr=None)  # Cargar el archivo de audio
        print(f"Archivo procesado cargado correctamente. Sample Rate: {sample_rate}")

        # Extraer características utilizando los datos del audio
        features = extractor.extraer_caracteristicas_generales(audio, sample_rate=sample_rate)  # Pasar datos cargados
        print(f"Características extraídas del archivo procesado: {features}")
        
        # Aplicar PCA si está habilitado
        if extractor.pca:
            features = extractor.pca.transform([features])[0]
            print(f"Características transformadas por PCA: {features}")

    except Exception as e:
        print(f"Error al procesar el archivo de prueba: {e}")
        return

if __name__ == "__main__":
    main()
