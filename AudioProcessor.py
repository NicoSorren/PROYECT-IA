
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

class AudioProcessor:
    def __init__(self, input_folder="AudiosOriginales", output_folder="AudiosProcesados", silence_threshold=30):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.silence_threshold = silence_threshold

        os.makedirs(self.output_folder, exist_ok=True)

    def reducir_ruido(self, audio, sample_rate):
        """Reducción de ruido usando noisereduce"""
        return nr.reduce_noise(y=audio, sr=sample_rate)

    def normalizar_audio(self, audio):
        """Normalizar el audio para que tenga una amplitud entre -1 y 1"""
        return audio / np.abs(audio).max() if np.abs(audio).max() != 0 else audio

    def eliminar_silencios(self, archivo_audio):
        """Eliminar silencios y reducir ruido"""
        ruta_audio = os.path.join(self.input_folder, archivo_audio)
        try:
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
        except Exception as e:
            print(f"Error al cargar {archivo_audio}: {e}")
            return

        # Reducir ruido
        audio_reducido = self.reducir_ruido(audio, sample_rate)

        # Eliminar silencios
        intervalos = librosa.effects.split(audio_reducido, top_db=self.silence_threshold)
        audio_sin_silencio = []
        for inicio, fin in intervalos:
            audio_sin_silencio.extend(audio_reducido[inicio:fin])

        # Normalizar y guardar
        audio_normalizado = self.normalizar_audio(np.array(audio_sin_silencio))
        nombre_salida = f"procesado_{archivo_audio.split('.')[0]}.wav"
        ruta_salida = os.path.join(self.output_folder, nombre_salida)
        sf.write(ruta_salida, audio_normalizado, sample_rate)

    def procesar_todos_los_audios(self):
        """Procesar todos los audios en la carpeta de entrada"""
        for archivo in os.listdir(self.input_folder):
            if archivo.endswith((".wav", ".ogg")):
                self.eliminar_silencios(archivo)
        print("Preprocesamiento completado.")

if __name__ == "__main__":
    procesador = AudioProcessor()
    procesador.procesar_todos_los_audios()
