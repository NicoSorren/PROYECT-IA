from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
import os
import numpy as np

def main():
    # Paso 1: Procesar audios de entrenamiento en "AudiosProcesados"
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=9)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels = extractor.procesar_todos_los_audios()

    # Mostrar características extraídas de los audios procesados
    print("\nCaracterísticas extraídas de los audios de entrenamiento:")
    for feature, label in zip(features_entrenamiento, labels):
        print(f"{label}: {feature}")

    # Paso 2: Entrenar el modelo KNN con los audios de entrenamiento
    clasificador = KnnAlgorithm(k=4)  # Instancia del modelo KNN
    clasificador.fit(features_entrenamiento, labels)  # Entrenar modelo con las características extraídas

    # Guardar el modelo entrenado
    clasificador.save_model("knn_model.pkl")

    # Paso 3: Procesar el archivo de audio de prueba 'papa_prueba.ogg' utilizando AudioProcessor
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

    # Paso 4: Extraer características del archivo procesado utilizando el extractor ajustado
    print("\nProcesando el archivo de prueba...")
    extractor.input_folder = archivo_procesado  # Cambiamos el input_folder al archivo de prueba
    features_prueba, _ = extractor.procesar_todos_los_audios()  # Extraemos las características

    print("\nCaracterísticas del archivo de prueba transformadas por PCA:")
    print(f"features_entrenamiento es: {features_entrenamiento} y su tamaño es {len(features_entrenamiento)}")
    print(f"features_prueba es: {features_prueba} y su tamaño es {len(features_prueba)}")  # El último elemento es el archivo de prueba

    # Paso 5: Usar el modelo entrenado KNN para predecir la palabra
    prediccion = clasificador.predict(features_prueba[-1])  # Usamos las características del archivo de prueba
    print(f"\nLa palabra predicha es: {prediccion}")

if __name__ == "__main__":
    main()