import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
from flask import Flask, request, jsonify

class LiveAudioProcessor:
    def __init__(self, filename="TempAudios/audio.wav", duration=3, sample_rate=44100):
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate

    def grabar_audio(self):
        """
        Graba el audio en vivo durante el tiempo especificado y guarda el archivo en WAV.
        """
        print(f"Grabando audio en vivo durante {self.duration} segundos...")

        # Graba el audio en vivo
        audio = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()  # Esperar hasta que termine la grabación

        # Guarda el archivo en la carpeta TempAudios como audio.wav
        wav.write(self.filename, self.sample_rate, (audio * 32767).astype(np.int16))  # Convertir a formato WAV
        print(f"Audio grabado y guardado como {self.filename}")
    
    def eliminar_audio(self):
        """
        Elimina el archivo de audio después de la ejecución.
        """
        if os.path.exists(self.filename):
            os.remove(self.filename)
            print(f"Archivo {self.filename} eliminado.")
            
    def iniciar_servidor(self):
        """
        Inicia un servidor Flask que permite grabar el audio al hacer una petición HTTP.
        """
        app = Flask(__name__)

        @app.route('/grabar', methods=['GET'])
        def grabar():
            # Método para iniciar la grabación cuando se haga la petición
            self.grabar_audio()
            return jsonify({"mensaje": "Audio grabado exitosamente", "archivo": self.filename})

        @app.route('/eliminar', methods=['GET'])
        def eliminar():
            # Método para eliminar el archivo grabado
            self.eliminar_audio()
            return jsonify({"mensaje": "Archivo eliminado exitosamente"})
        
        app.run(host='0.0.0.0', port=5000, debug=True)

