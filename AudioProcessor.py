import os
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import noisereduce as nr
import numpy as np
from collections import defaultdict

class AudioProcessor:
    def __init__(self, input_folder="AudiosOriginales", output_folder="AudiosProcesados", silence_threshold=30):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.silence_threshold = silence_threshold
        self.zcr_results = defaultdict(lambda: {
            'zcr_promedios': [], 
            'zcr_maximos': [], 
            'amplitud_segmento_mayor': []  # Almacenar los segmentos con mayor amplitud
        })
        self.feature_matrix = []  # Para almacenar las características para PCA
        self.labels = []          # Para almacenar las etiquetas (nombre de la verdura) de cada muestra

        os.makedirs(self.output_folder, exist_ok=True)

    def aplicar_filtro_pasabanda(self, audio, sample_rate, lowcut=300, highcut=3400):
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(1, [low, high], btype='band')
        audio_filtrado = lfilter(b, a, audio)
        return audio_filtrado

    def reducir_ruido(self, audio, sample_rate):
        audio_reducido = nr.reduce_noise(y=audio, sr=sample_rate)
        return audio_reducido

    def normalizar_audio(self, audio):
        return audio / np.abs(audio).max()

    def calcular_zcr_amplitud_segmentos(self, audio, sample_rate, num_segmentos=10):
        duracion_segmento = len(audio) // num_segmentos
        zcr_promedios = []
        zcr_maximos = []

        amplitud_segmento_mayor = 0
        segmento_con_mayor_amplitud = 0

        for i in range(num_segmentos):
            inicio = i * duracion_segmento
            fin = inicio + duracion_segmento if i < num_segmentos - 1 else len(audio)
            segmento = audio[inicio:fin]

            # Calcular ZCR del segmento
            zcr = librosa.feature.zero_crossing_rate(segmento)[0]
            zcr_promedios.append(np.mean(zcr))
            zcr_maximos.append(np.max(zcr))

            # Calcular amplitud máxima del segmento
            amplitud_maxima = np.max(np.abs(segmento))

            # Identificar el segmento con la mayor amplitud
            if amplitud_maxima > amplitud_segmento_mayor:
                amplitud_segmento_mayor = amplitud_maxima
                segmento_con_mayor_amplitud = i + 1  # +1 para contar desde 1

        # Calcular valores globales para ZCR y registrar el segmento con mayor amplitud
        caracteristicas = {
            'zcr_promedio_global': np.mean(zcr_promedios),
            'zcr_maximo_global': np.max(zcr_maximos),
            'segmento_con_mayor_amplitud': segmento_con_mayor_amplitud
        }

        return caracteristicas

    def eliminar_silencios(self, archivo_audio):
        ruta_audio = os.path.join(self.input_folder, archivo_audio)
        try:
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
        except Exception as e:
            print(f"Error al cargar {archivo_audio}: {e}")
            return

        audio_reducido = self.reducir_ruido(audio, sample_rate)
        audio_filtrado = self.aplicar_filtro_pasabanda(audio_reducido, sample_rate)

        intervalos = librosa.effects.split(audio_filtrado, top_db=self.silence_threshold)

        audio_sin_silencio = []
        for inicio, fin in intervalos:
            audio_sin_silencio.extend(audio_filtrado[inicio:fin])

        # Normalizar el audio antes de guardar
        audio_normalizado = self.normalizar_audio(np.array(audio_sin_silencio))

        # Calcular características de ZCR y amplitud
        caracteristicas = self.calcular_zcr_amplitud_segmentos(audio_normalizado, sample_rate)

        # Obtener el nombre de la verdura a partir del nombre del archivo (se asume formato "verdura_numero")
        nombre_verdura = archivo_audio.split("_")[0]

        # Almacenar los resultados en el diccionario y feature_matrix
        self.zcr_results[nombre_verdura]['zcr_promedios'].append(caracteristicas['zcr_promedio_global'])
        self.zcr_results[nombre_verdura]['zcr_maximos'].append(caracteristicas['zcr_maximo_global'])
        self.zcr_results[nombre_verdura]['amplitud_segmento_mayor'].append(caracteristicas['segmento_con_mayor_amplitud'])
        
        # Guardar las características relevantes para PCA en feature_matrix
        self.feature_matrix.append([
            caracteristicas['zcr_promedio_global'],
            caracteristicas['zcr_maximo_global'],
            caracteristicas['segmento_con_mayor_amplitud']
        ])
        self.labels.append(nombre_verdura)

        nombre_salida = f"procesado_{archivo_audio.split('.')[0]}.wav"
        ruta_salida = os.path.join(self.output_folder, nombre_salida)
        sf.write(ruta_salida, audio_normalizado, sample_rate)

    def procesar_todos_los_audios(self):
        for archivo in os.listdir(self.input_folder):
            if archivo.endswith((".wav", ".ogg")):
                self.eliminar_silencios(archivo)

        # Calcular e imprimir los promedios finales por verdura, incluyendo el segmento promedio con mayor amplitud
        for verdura, datos in self.zcr_results.items():
            promedio_final_zcr = np.mean(datos['zcr_promedios'])
            maximo_final_zcr = np.mean(datos['zcr_maximos'])
            segmento_promedio_mayor_amplitud = np.mean(datos['amplitud_segmento_mayor'])  # Promedio de los segmentos con mayor amplitud

            print(f"Verdura: {verdura} - ZCR Promedio Final: {promedio_final_zcr}, ZCR Máximo Final: {maximo_final_zcr}")
            print(f"Segmento Promedio con Mayor Amplitud: {segmento_promedio_mayor_amplitud:.2f}")

# Ejemplo de uso:
input_folder = "AudiosOriginales"  
output_folder = "AudiosProcesados"

procesador = AudioProcessor(input_folder, output_folder, silence_threshold=30)
procesador.procesar_todos_los_audios()
