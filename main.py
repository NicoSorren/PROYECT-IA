from FeatureExtractor import FeatureExtractor
from AudioClassifierKNN import AudioClassifierKNN
import os

# Paso 1: Extraer características usando FeatureExtractor
extractor = FeatureExtractor(use_pca=True, n_components=9)
features, labels = extractor.procesar_todos_los_audios()

# Obtener los nombres de los archivos procesados
nombres_archivos = [f"procesado_{archivo}" for archivo in os.listdir("AudiosProcesados") if archivo.endswith(".wav")]

# Visualizar las características en 3D con etiquetas
extractor.visualizar_caracteristicas_3d_con_etiquetas(nombres_archivos)

# Paso 2: Entrenar y evaluar el modelo KNN
clasificador = AudioClassifierKNN(features, labels)
clasificador.entrenar_modelo()
