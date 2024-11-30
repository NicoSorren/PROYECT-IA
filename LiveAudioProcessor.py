import librosa
import numpy as np
import os
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
from sklearn.decomposition import PCA  # Asegurarnos de que PCA esté importado

class LiveAudioProcessor:
    def __init__(self, input_folder="AudiosOriginales", output_folder="TempAudios"):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Verificar si la carpeta de salida existe, si no crearla
        os.makedirs(self.output_folder, exist_ok=True)

    def procesar_audio(self, archivo_audio):
        """Procesar el archivo de audio de prueba, extrayendo las características"""
        print(f"Procesando el archivo: {archivo_audio}")
        try:
            # Cargar el archivo de audio
            audio, sample_rate = librosa.load(archivo_audio, sr=None)
            print(f"Archivo cargado correctamente. Sample Rate: {sample_rate}")
        except Exception as e:
            print(f"Error al cargar {archivo_audio}: {e}")
            return None

        # Extraer características usando FeatureExtractor
        extractor = FeatureExtractor(use_pca=True, n_components=9)
        features = extractor.extraer_caracteristicas_generales(audio, sample_rate)

        # Asegurarse de que las características sean del formato adecuado para la predicción
        features = np.array(features).reshape(1, -1)  # Reshape para tener la forma adecuada
        
        # Si PCA está habilitado, aplicamos PCA (asegurándonos de que PCA esté disponible)
        if extractor.use_pca:
            # Si no se ha ajustado el PCA antes, entrenarlo ahora
            if not hasattr(extractor, 'pca'):
                print("Entrenando PCA con el archivo de prueba...")
                pca = PCA(n_components=extractor.n_components)
                features = pca.fit_transform(features)  # Aplicar fit_transform al nuevo audio
                extractor.pca = pca  # Guardar el modelo PCA entrenado
            else:
                features = extractor.pca.transform(features)  # Aplicar PCA usando el modelo entrenado

        print(f"Características extraídas: {features.shape}")
        return features

    def predecir_palabra(self, features):
        """Predecir la palabra utilizando el modelo KNN entrenado"""
        try:
            # Cargar el modelo KNN previamente entrenado
            clasificador = KnnAlgorithm.load_model()  # Cargar el modelo
            prediccion = clasificador.predict(features)
            print(f"La palabra predicha es: {prediccion}")
        except Exception as e:
            print(f"Error al cargar el modelo o realizar la predicción: {e}")

# Ejecución del flujo con el archivo de prueba
if __name__ == "__main__":
    # Paso 1: Cargar el archivo de audio que contiene la palabra "zanahoria"
    archivo_audio = "zanahoria_prueba.ogg"  # Nombre del archivo de audio de prueba

    # Verificar que el archivo existe en el directorio
    if not os.path.exists(archivo_audio):
        print(f"Error: El archivo {archivo_audio} no existe.")
    else:
        # Paso 2: Procesar el audio y extraer las características
        live_processor = LiveAudioProcessor()
        features = live_processor.procesar_audio(archivo_audio)

        if features is not None:
            # Paso 3: Usar el modelo para predecir la palabra
            live_processor.predecir_palabra(features)
