from FeatureExtractor import FeatureExtractor
import streamlit as st
import os
from AudioProcessor import AudioProcessor

def procesar_entrenamiento():
        """
        Procesa los audios de entrenamiento y devuelve características y etiquetas.
        """
        st.write("Procesando audios de entrenamiento... Esto puede tardar un momento.")
        extractor = FeatureExtractor(input_folder="AudiosProcesados", use_pca=True, n_components=9)

        features_entrenamiento, labels, _ = extractor.procesar_todos_los_audios()
        
        st.success("Procesamiento de entrenamiento completado.")
        return features_entrenamiento, labels, extractor

def mostrar_resumen(features_entrenamiento, labels):
        """
        Muestra un resumen de los datos procesados, excluyendo la matriz de características.
        """
        st.subheader("Resumen de los audios procesados")
        
        if len(features_entrenamiento) > 0:
            st.write(f"Cantidad de audios procesados: {len(features_entrenamiento)}")
            st.write("Etiquetas disponibles:")
            etiquetas = set(labels)
            for etiqueta in etiquetas:
                st.write(f"- {etiqueta}: {labels.count(etiqueta)} muestras")
        else:
            st.error("No se encontraron audios procesados.")

def procesar_evaluar_audio(archivo_procesado, extractor, clasificador):
    st.write("Procesando archivo de prueba...")  
    
    if os.path.exists(archivo_procesado):
            st.success(f"Archivo procesado correctamente: {archivo_procesado}")
            
            # Extraer características
            extractor.input_folder = archivo_procesado
            _, _, features_prueba = extractor.procesar_todos_los_audios()
            
            # Predicción del modelo
            prediccion = clasificador.predict(features_prueba)
            st.subheader("Resultado de la Predicción")
            st.write(f"La clase predicha para el archivo de prueba es: **{prediccion}**")
    else:
            st.error("Error al procesar el archivo de prueba. Asegúrate de que TempAudios contiene el archivo correcto.")
