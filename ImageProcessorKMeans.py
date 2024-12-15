from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para el grafico 3D

class ImageProcessorKMeans:
    def __init__(self, image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=3):
        self.image_folder = image_folder
        self.segmented_folder = segmented_folder
        self.k = k
        os.makedirs(self.segmented_folder, exist_ok=True)

    def aplicar_kmeans(self, imagen):
        """
        Aplica K-Means para segmentar la imagen en K clusters.
        """
        original_shape = imagen.shape
        imagen_reshape = imagen.reshape((-1, 3))

        # Aplicar K-Means
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(imagen_reshape)
        labels = kmeans.predict(imagen_reshape)
        colores_centrales = kmeans.cluster_centers_

        # Reconstruir la imagen segmentada
        imagen_segmentada = colores_centrales[labels].reshape(original_shape)
        return imagen_segmentada.astype(np.uint8)

    def procesar_y_guardar_segmentadas(self):
        """
        Aplica K-Means a todas las imágenes y las guarda.
        """
        for verdura in os.listdir(self.image_folder):
            ruta_verdura = os.path.join(self.image_folder, verdura)
            if os.path.isdir(ruta_verdura):
                carpeta_destino = os.path.join(self.segmented_folder, verdura)
                os.makedirs(carpeta_destino, exist_ok=True)

                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)
                    if imagen is not None:
                        # Aplicar K-Means
                        imagen_segmentada = self.aplicar_kmeans(imagen)

                        # Guardar la imagen segmentada
                        ruta_guardado = os.path.join(carpeta_destino, f"segmentada_{imagen_nombre}")
                        cv2.imwrite(ruta_guardado, cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR))
                        print(f"Imagen segmentada guardada: {ruta_guardado}")

    def extraer_caracteristicas_color(self):
        """
        Extrae las características promedio RGB de las imágenes segmentadas.
        """
        caracteristicas = {}
        for verdura in os.listdir(self.segmented_folder):
            ruta_verdura = os.path.join(self.segmented_folder, verdura)
            if os.path.isdir(ruta_verdura):
                caracteristicas[verdura] = []
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)
                    if imagen is not None:
                        promedio_color = np.mean(imagen, axis=(0, 1))  # Promedio RGB
                        caracteristicas[verdura].append(promedio_color)
        return caracteristicas

    def graficar_caracteristicas(self, caracteristicas):
        """
        Grafica las características RGB en un espacio 3D.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colores = {'berenjena': 'purple', 'zanahoria': 'orange', 'papa': 'yellow', 'batata': 'brown'}

        for verdura, lista_colores in caracteristicas.items():
            lista_colores = np.array(lista_colores)
            r, g, b = lista_colores[:, 0], lista_colores[:, 1], lista_colores[:, 2]
            ax.scatter(r, g, b, label=verdura, color=colores.get(verdura, 'black'))

        ax.set_xlabel('Rojo')
        ax.set_ylabel('Verde')
        ax.set_zlabel('Azul')
        ax.set_title('Distribución de colores promedio por verdura')
        ax.legend()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessorKMeans(image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=3)
    procesador.procesar_y_guardar_segmentadas()
    
    # Extraer características
    caracteristicas = procesador.extraer_caracteristicas_color()
    
    # Graficar características
    procesador.graficar_caracteristicas(caracteristicas)
