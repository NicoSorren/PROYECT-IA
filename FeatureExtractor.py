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

    def procesar_audio(self, archivo_audio):
        ruta_audio = os.path.join(self.input_folder, archivo_audio)
        try:
            audio, sample_rate = librosa.load(ruta_audio, sr=None)
        except Exception as e:
            print(f"Error al cargar {archivo_audio}: {e}")
            return

        energia = self.calcular_energia(audio)
        nombre_verdura = archivo_audio.split("_")[1]

        if nombre_verdura in ['berenjena', 'zanahoria']:
            caracteristicas = self.extraer_caracteristicas_berenjena_zanahoria(audio, sample_rate)
        else:
            caracteristicas = self.extraer_caracteristicas_generales(audio, sample_rate)
        
        features = [energia] + list(caracteristicas)
        
        if len(features) < self.feature_length:
            features.extend([0] * (self.feature_length - len(features)))
        elif len(features) > self.feature_length:
            features = features[:self.feature_length]

        self.feature_matrix.append(features)
        self.labels.append(nombre_verdura)
        print(f"Audio: {archivo_audio} - Energía: {energia:.4f} - Características extraídas")

    def procesar_todos_los_audios(self):
        for archivo in os.listdir(self.input_folder):
            if archivo.endswith(".wav"):
                self.procesar_audio(archivo)
        
        self.feature_matrix = np.array(self.feature_matrix)
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(self.feature_matrix)
        
        if self.use_pca:
            pca = PCA(n_components=self.n_components)
            self.feature_matrix = pca.fit_transform(self.feature_matrix)
            print("PCA aplicado. Componentes retenidos:", self.n_components)
        
        return self.feature_matrix, self.labels

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

