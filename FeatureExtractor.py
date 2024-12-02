import os
import librosa
import numpy as np
import scipy.signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FeatureExtractor:
    def __init__(self, input_folder="AudiosProcesados", use_pca=True, n_components=3):
        self.input_folder = input_folder
        self.feature_matrix = []
        self.labels = []
        self.use_pca = use_pca
        self.n_components = n_components
        self.feature_length = 15
        self.pca = None  # Inicializamos PCA como None
        self.scaler = StandardScaler()  # Inicializamos el scaler. # StandardScaler estandariza los datos a una distribución con media 0 y desviación estándar 1.
        self.feature_prueba = []

    def calcular_energia(self, audio):
        return np.sum(audio**2) / len(audio)

    def calcular_formantes(self, audio_segment, sample_rate):
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_segment[0], audio_segment[1:] - pre_emphasis * audio_segment[:-1])
        hamming_window = np.hamming(len(emphasized_audio))
        fft_spectrum = np.fft.fft(emphasized_audio * hamming_window)
        freqs = np.fft.fftfreq(len(fft_spectrum))
        positive_freqs = freqs[:len(freqs) // 2]
        magnitude_spectrum = np.abs(fft_spectrum[:len(fft_spectrum) // 2])
        peaks, _ = scipy.signal.find_peaks(magnitude_spectrum, height=0)
        formants = positive_freqs[peaks] * sample_rate
        return formants[:2] if len(formants) >= 2 else [0, 0]

    def extraer_caracteristicas_berenjena_zanahoria(self, audio, sample_rate, n_mfcc=13):
        duracion_segmento = len(audio) // 3
        inicio = audio[:duracion_segmento]
        mfcc = librosa.feature.mfcc(y=inicio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        formantes = self.calcular_formantes(inicio, sample_rate)
        features = np.concatenate((mfcc_mean, formantes))
        return features

    def extraer_caracteristicas_generales(self, audio, sample_rate, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        features = np.concatenate((mfcc_mean, [zcr_mean]))
        return features
    
    def procesar_todos_los_audios(self):
        """Este método ahora se puede usar tanto para un directorio de audios como para un solo archivo"""
        print(f"Verificando la ruta de entrada: {self.input_folder}")

        if os.path.isdir(self.input_folder):    # verifica si input_folder es un directorio
            # Listar todos los archivos en la carpeta para asegurarse de que están ahí
            archivos_en_directorio = os.listdir(self.input_folder)
            if not archivos_en_directorio:
                print("¡Advertencia! El directorio está vacío.")

            for archivo in os.listdir(self.input_folder):
                if archivo.endswith(".wav"):
                    self.procesar_audio(archivo)
            
            self.feature_matrix = np.array(self.feature_matrix)     # convierte self.feature_matrix (que hasta este punto es una lista de listas) en un array de NumPy.
            print(f"Características extraídas de {len(self.feature_matrix)} archivos.")

            self.feature_matrix = self.scaler.fit_transform(self.feature_matrix) # fit_transform ajusta el scaler a los datos (calcula media y desviación estándar) y luego aplica la transformación

            if self.use_pca:
                self.pca = PCA(n_components=self.n_components)      # Si self.use_pca es True, aplica el algoritmo PCA para reducir el número de características (dimensiones) de los datos. n_components determina el número de dimensiones retenidas.
                self.feature_matrix = self.pca.fit_transform(self.feature_matrix)   # fit_transform ajusta el PCA a los datos y los transforma, produciendo una nueva matriz donde cada fila representa un audio y cada columna representa un componente principal.
                print("PCA aplicado. Componentes retenidos:", self.n_components)
        
        else:
            try:
                print(f"input_folder es: {self.input_folder}")
                self.feature_prueba = self.procesar_audio(self.input_folder)
            except Exception as e:
                print("No se pudo ejecutar procesar_audios con audio de prueba")
                
                                
        return self.feature_matrix, self.labels, self.feature_prueba

    def procesar_audio(self, archivo_audio):
        if os.path.isdir(self.input_folder):
            ruta_audio = os.path.join(self.input_folder, archivo_audio)  # Ruta para los audios en carpeta
        else:
            ruta_audio = self.input_folder  # Ruta directa para el archivo de prueba
        try:
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
        except Exception as e:
            print(f"Error al cargar {ruta_audio}: {e}")

        energia = self.calcular_energia(audio)   

        if os.path.isdir(self.input_folder):
            nombre_verdura = archivo_audio.split("_")[1]  # Esto puede necesitar ajustes dependiendo del formato de nombre de archivo
            if nombre_verdura in ['berenjena', 'zanahoria']:
                caracteristicas = self.extraer_caracteristicas_berenjena_zanahoria(audio, sample_rate)
            else:
                caracteristicas = self.extraer_caracteristicas_generales(audio, sample_rate)
        else:
            try:
                caracteristicas = self.extraer_caracteristicas_generales(audio, sample_rate)
            except Exception as e:
                print(f"no se pudo ejecutar extraer_caracteristicas_generales con archivo de prueba")
            
        
        features = [energia] + list(caracteristicas)
        """Esta línea de código está ajustando la longitud del vector de características (features) para que tenga exactamente la longitud esperada, definida por self.feature_length
        """
        if len(features) < self.feature_length:
            features.extend([0] * (self.feature_length - len(features)))    #Se calcula cuántos elementos faltan: self.feature_length - len(features) y se Se añaden ceros ([0]) al final del vector para completar la longitud requerida.
        elif len(features) > self.feature_length:
            features = features[:self.feature_length]   # Se recortan los elementos sobrantes para que solo queden los primeros self.feature_length.

        if os.path.isdir(self.input_folder):
            self.feature_matrix.append(features)
            self.labels.append(nombre_verdura)
            
        else:
            # Para el archivo de prueba, devolvemos las características directamente
            try:
                features = np.array(features).reshape(1, -1)  # Reshape para tener la forma adecuada
            except Exception as e:
                print("No se ha podido aplicar np.array")

            try:
                if not hasattr(self.scaler, 'mean_'):
                    raise ValueError("El escalador no está ajustado. Asegúrate de procesar los datos de entrenamiento primero.")
                features = self.scaler.transform(features)
            except Exception as e:
                print(f"No se ha podido aplicar transform: {e}")
            
            if self.pca:
                try:
                    features = self.pca.transform(features)  # Si se aplica PCA, también se transforma
                except Exception as e:
                    print("No se ha podido aplicar PCA")
            print(f"Características extraídas y transformadas del archivo de prueba: {features}")
           
        return features


    def visualizar_caracteristicas_3d_con_etiquetas(self, nombres_archivos):
        etiquetas = set(self.labels)
        colores = ['blue', 'orange', 'green', 'red']
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        for etiqueta, color in zip(etiquetas, colores):
            indices = [i for i, label in enumerate(self.labels) if label == etiqueta]
            for idx in indices:
                ax.scatter(
                    self.feature_matrix[idx, 0], 
                    self.feature_matrix[idx, 1], 
                    self.feature_matrix[idx, 2], 
                    label=etiqueta if idx == indices[0] else "", 
                    c=color, alpha=0.7
                )
                # Añadimos la etiqueta del archivo en cada punto
                ax.text(self.feature_matrix[idx, 0], 
                        self.feature_matrix[idx, 1], 
                        self.feature_matrix[idx, 2], 
                        nombres_archivos[idx], fontsize=8)
        
        ax.set_title("Características de Audio - PCA 3D con Etiquetas")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.set_zlabel("Componente principal 3")
        ax.legend()
        plt.show()
