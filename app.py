import streamlit as st
import os
from AudioProcessor import AudioProcessor
from FeatureExtractor import FeatureExtractor
from KnnAlgorithm import KnnAlgorithm
from MetodosApp import procesar_entrenamiento, mostrar_resumen, procesar_evaluar_audio

# Configuración de la aplicación
st.title("Clasificador de Audio")
st.sidebar.title("Menú")

# Opciones en la barra lateral
menu = ["Entrenamiento", "Procesar audio grabado"]
selected = st.sidebar.radio("Selecciona una opción", menu)

# Parámetros de configuración
input_folder = "AudiosProcesados"

archivo_prueba = "audio_prueba.ogg"
procesador = AudioProcessor(input_folder="TempAudios", output_folder="TempAudios")

# Instancias globales para entrenamiento
extractor = FeatureExtractor(input_folder="AudiosProcesador", use_pca=True, n_components=9)
clasificador = KnnAlgorithm(k=4)


if selected == "Entrenamiento":
    st.subheader("Entrenamiento del Modelo")
    # Botón para procesar los audios

    if st.button("Procesar audios de entrenamiento"):
        features_entrenamiento, labels, extractor = procesar_entrenamiento()
        mostrar_resumen(features_entrenamiento, labels)

    if st.button("Entrenar modelo Knn"):# Entrenar el modelo KNN
        st.write("Entrenando modelo KNN...")
        features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()
        clasificador.fit(features_entrenamiento, labels)
        clasificador.save_model("knn_model.pkl")
        clasificador.evaluate(features_entrenamiento, labels)  # Evaluamos con los mismos datos de entrenamiento


        st.success("Modelo entrenado y guardado exitosamente en el repositorio como knn_model.pkl ")
        
elif selected == "Procesar audio grabado":
    st.subheader("Procesar y Evaluar Audio Grabado")

    if st.button("Procesar audio"):
        st.write("Procesando audio...")
        procesador.eliminar_silencios(archivo_prueba)
        
        archivo_procesado = "TempAudios/procesado_audio_prueba.wav"
        
        procesar_evaluar_audio(archivo_procesado, extractor, clasificador)
    
        st.success("El audio se ha evaluado con éxito")

