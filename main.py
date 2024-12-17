from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ImageProcessorKMeans import ImageProcessorKMeans

def main():



    # Paso 1: Procesar audios de entrenamiento en "AudiosProcesados"
    extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=9)
    print("\nProcesando audios de entrenamiento...")
    features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()

    # Paso 2: Entrenar el modelo KNN con los audios de entrenamiento
    clasificador = KnnAlgorithm(k=4)  # Instancia del modelo KNN
    clasificador.fit(features_entrenamiento, labels)  # Entrenar modelo con las características extraídas

    # Guardar el modelo entrenado
    clasificador.save_model("knn_model.pkl")
    clasificador.evaluate(features_entrenamiento, labels)  # Evaluamos con los mismos datos de entrenamiento

    carpeta = "C:/Users/nsorr/Desktop/PROJECT IA/TempAudios"
    
    for archivo in os.listdir(carpeta):
        if archivo.startswith("WhatsApp") and archivo.endswith(".ogg"):
            nuevo_nombre = f"audio.ogg"
            try:
                os.rename(os.path.join(carpeta, archivo), os.path.join(carpeta, nuevo_nombre))
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")
            except OSError as e:
                print(f"Error al renombrar {archivo}: {e}")

            
    # Paso 3: Procesar el archivo de audio de prueba 'papa_prueba.ogg' utilizando AudioProcessor
    archivo_audio = "audio.ogg"
    procesador = AudioProcessor(input_folder="TempAudios", output_folder="TempAudios")
    procesador.eliminar_silencios(archivo_audio)  # Procesar el archivo y almacenarlo en TempAudios como .wav

    # Verificar si el archivo procesado está disponible
    archivo_procesado = "TempAudios/procesado_audio.wav"
    if os.path.exists(archivo_procesado):
        print(f"El archivo procesado se ha guardado correctamente como {archivo_procesado}")
    else:
        print(f"Error: El archivo procesado {archivo_procesado} no se ha guardado correctamente.")
        return

    # Paso 4: Extraer características del archivo procesado utilizando el extractor ajustado
    print("\nProcesando el archivo de prueba...")
    extractor.input_folder = archivo_procesado  # Cambiamos el input_folder al archivo de prueba
    _, _, features_prueba = extractor.procesar_todos_los_audios()  # Extraemos las características

    print("\nCaracterísticas del archivo de prueba transformadas por PCA:")

    # Paso 5: Usar el modelo entrenado KNN para predecir la palabra
    prediccion = clasificador.predict(features_prueba)  # Usamos las características del archivo de prueba
    print(f"\nLa palabra predicha es: {prediccion}")

    """Elimina todos los archivos dentro de una carpeta."""
    carpeta = "TempAudios"
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        try:
            os.remove(ruta_archivo)  # Eliminar el archivo
        except Exception as e:
            print(f"No se pudo eliminar {ruta_archivo}: {e}")

    # Paso 6: Mostrar la imagen correspondiente a la predicción
    carpeta_imagenes_etiquetadas = "ImagenesEtiquetadas"
    nombre_imagen = f"{prediccion}.jpg"  # Buscar la imagen con el nombre de la palabra predicha
    ruta_imagen = os.path.join(carpeta_imagenes_etiquetadas, nombre_imagen)

    if os.path.exists(ruta_imagen):
        print(f"Mostrando imagen correspondiente a la predicción: {ruta_imagen}")
        # Cargar y mostrar la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is not None:
            plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
            plt.title(f"Predicción: {prediccion}")
            plt.axis("off")
            plt.show()
        else:
            print(f"No se pudo cargar la imagen {ruta_imagen}")
    else:
        print(f"No se encontró la imagen etiquetada para la predicción '{prediccion}'.")

if __name__ == "__main__":
    main()