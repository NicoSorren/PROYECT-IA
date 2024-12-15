import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageProcessor:
    def __init__(self, image_folder="ImagenesVerduras", processed_folder="ImagenesProcesadas"):
        self.image_folder = image_folder
        self.processed_folder = processed_folder
        os.makedirs(self.processed_folder, exist_ok=True)  # Crear la carpeta si no existe

    def cargar_imagenes(self):
        """
        Carga las imágenes desde las carpetas organizadas por verduras.
        """
        imagenes = {}
        for verdura in os.listdir(self.image_folder):
            ruta_verdura = os.path.join(self.image_folder, verdura)
            if os.path.isdir(ruta_verdura):
                imagenes[verdura] = []
                for imagen_nombre in os.listdir(ruta_verdura):
                    ruta_imagen = os.path.join(ruta_verdura, imagen_nombre)
                    imagen = cv2.imread(ruta_imagen)  # Cargar la imagen
                    if imagen is not None:
                        imagenes[verdura].append((imagen_nombre, imagen))  # Guardar nombre e imagen
                    else:
                        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        return imagenes

    def aplicar_transformaciones(self, imagen):
        """
        Aplica transformaciones a la imagen: exposición, contraste, saturación, y nitidez.
        """
        # Aumentar el contraste
        alpha = 1.3 # Factor de contraste (>1 aumenta el contraste)
        beta = 10    # Valor de brillo
        imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

        # Aumentar la saturación
        hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 30)  # Aumentar el canal de saturación
        imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Redimensionar la imagen a 224x224
        imagen = cv2.resize(imagen, (224, 224), interpolation=cv2.INTER_AREA)

        # Filtro de nitidez
        kernel = np.array([[0, -1, 0], [-1, 5.3, -1], [0, -1, 0]])
        imagen = cv2.filter2D(imagen, -1, kernel)

        return imagen

    def procesar_y_guardar(self, imagenes):
        """
        Aplica transformaciones y guarda las imágenes en la carpeta ImagenesProcesadas.
        """
        for verdura, lista_imagenes in imagenes.items():
            carpeta_destino = os.path.join(self.processed_folder, verdura)
            os.makedirs(carpeta_destino, exist_ok=True)
            for nombre, imagen in lista_imagenes:
                # Aplicar transformaciones
                imagen_procesada = self.aplicar_transformaciones(imagen)
                # Guardar imagen procesada
                ruta_guardado = os.path.join(carpeta_destino, nombre)
                cv2.imwrite(ruta_guardado, cv2.cvtColor(imagen_procesada, cv2.COLOR_RGB2BGR))
                print(f"Imagen guardada: {ruta_guardado}")

    def mostrar_imagenes(self, imagenes, num_por_clase=3, procesadas=True):
        """
        Muestra un número específico de imágenes por clase (verdura).
        Si procesadas es True, aplica transformaciones antes de mostrarlas.
        """
        clases = list(imagenes.keys())
        total_clases = len(clases)
        
        plt.figure(figsize=(15, 5 * total_clases))  # Ajustar el tamaño de la figura para mostrar todas las clases
        
        for verdura, lista_imagenes in imagenes.items():
            print(f"Mostrando imágenes de: {verdura}")
            plt.figure(figsize=(15, 5))
            for i, (_, imagen) in enumerate(lista_imagenes[:num_por_clase]):
                if procesadas:
                    imagen = self.aplicar_transformaciones(imagen)
                plt.subplot(1, num_por_clase, i + 1)
                plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                plt.title(f"{verdura} - {i + 1}")
                plt.axis("off")
            plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessor(image_folder="ImagenesVerduras", processed_folder="ImagenesProcesadas")
    imagenes = procesador.cargar_imagenes()
    procesador.mostrar_imagenes(imagenes, num_por_clase=3)
    procesador.procesar_y_guardar(imagenes)