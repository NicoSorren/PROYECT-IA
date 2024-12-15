from sklearn.cluster import KMeans
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para el grafico 3D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ImageProcessor import ImageProcessor

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

    def extraer_caracteristicas_color(self, folder):
        """
        Extrae las características promedio RGB de las imágenes en una carpeta específica.
        """
        caracteristicas = []
        etiquetas = []
        for verdura in os.listdir(folder):
            ruta_verdura = os.path.join(folder, verdura)
            if os.path.isdir(ruta_verdura):
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)
                    if imagen is not None:
                        promedio_color = np.mean(imagen, axis=(0, 1))  # Promedio RGB
                        caracteristicas.append(promedio_color)
                        etiquetas.append(verdura)
        return np.array(caracteristicas), np.array(etiquetas)

    def entrenar_y_evaluar(self):
        """
        Divide el dataset en entrenamiento y evaluación, entrena KMeans y calcula precisión.
        """
        # Extraer características y etiquetas del dataset
        caracteristicas, etiquetas = self.extraer_caracteristicas_color(self.segmented_folder)
        
        # Dividir en conjunto de entrenamiento y evaluación
        X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.3, random_state=42)

        # Aplicar KMeans al conjunto de entrenamiento
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans.fit(X_train)

        # Asignar etiquetas a clusters basados en la mayoría
        etiquetas_clusters = {}
        for i in range(self.k):
            indices_cluster = np.where(kmeans.labels_ == i)
            etiquetas_reales = y_train[indices_cluster]
            etiqueta_mayoritaria = max(set(etiquetas_reales), key=list(etiquetas_reales).count)
            etiquetas_clusters[i] = etiqueta_mayoritaria

        # Evaluar el conjunto de evaluación
        predicciones_clusters = kmeans.predict(X_test)
        predicciones = [etiquetas_clusters[cluster] for cluster in predicciones_clusters]

        # Calcular precisión
        precision = accuracy_score(y_test, predicciones)
        print(f"Precisión del modelo KMeans: {precision:.2f}")

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

    def predecir_imagen_nueva(self, temp_folder):
        """
        Carga una imagen nueva, la preprocesa, segmenta y predice la verdura.
        """
        # Verificar si la carpeta está vacía
        if not os.listdir(temp_folder):
            print(f"Error: La carpeta '{temp_folder}' está vacía. Agrega una imagen para evaluar.")
            return

        # Instanciar ImageProcessor para reutilizar los métodos de preprocesamiento
        procesador = ImageProcessor()

        for imagen_nombre in os.listdir(temp_folder):
            ruta_imagen = os.path.join(temp_folder, imagen_nombre)
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
                print(f"Evaluando imagen: {imagen_nombre}")
                
                # Preprocesar imagen usando ImageProcessor
                imagen_procesada = procesador.aplicar_transformaciones(imagen)
                
                # Aplicar K-Means para segmentar la imagen
                imagen_segmentada = self.aplicar_kmeans(imagen_procesada)

                # Extraer características
                promedio_color = np.mean(imagen_segmentada, axis=(0, 1))

                # Cargar características globales y entrenar modelo
                caracteristicas, etiquetas = self.extraer_caracteristicas_color(self.segmented_folder)
                kmeans = KMeans(n_clusters=self.k, random_state=0)
                kmeans.fit(caracteristicas)

                # Predecir cluster de la imagen nueva
                cluster = kmeans.predict([promedio_color])[0]

                # Asignar etiqueta según cluster
                etiquetas_clusters = {}
                for i in range(self.k):
                    indices_cluster = np.where(kmeans.labels_ == i)
                    etiquetas_reales = etiquetas[indices_cluster]
                    etiqueta_mayoritaria = max(set(etiquetas_reales), key=list(etiquetas_reales).count)
                    etiquetas_clusters[i] = etiqueta_mayoritaria

                prediccion = etiquetas_clusters[cluster]
                print(f"La imagen fue clasificada como: {prediccion}")

                # Mostrar imagen segmentada 
                plt.imshow(cv2.cvtColor(imagen_segmentada, cv2.COLOR_BGR2RGB))
                plt.title(f"Predicción: {prediccion}")
                plt.axis("off")
                plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessorKMeans(image_folder="ImagenesProcesadas", segmented_folder="ImagenesSegmentadas", k=4)
    procesador.procesar_y_guardar_segmentadas()
    procesador.entrenar_y_evaluar()
    procesador.predecir_imagen_nueva(temp_folder="TempImagenes")
