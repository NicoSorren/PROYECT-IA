import os
import cv2
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_folder="ImagenesVerduras"):
        self.image_folder = image_folder

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
                        imagenes[verdura].append(imagen)
                    else:
                        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        return imagenes

    def mostrar_imagenes(self, imagenes, num_por_clase=3):
        """
        Muestra un número específico de imágenes por clase (verdura).
        """
        for verdura, lista_imagenes in imagenes.items():
            print(f"Mostrando imágenes de: {verdura}")
            plt.figure(figsize=(15, 5))
            for i, imagen in enumerate(lista_imagenes[:num_por_clase]):
                plt.subplot(1, num_por_clase, i + 1)
                plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                plt.title(f"{verdura} - {i + 1}")
                plt.axis("off")
            plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    procesador = ImageProcessor(image_folder="ImagenesVerduras")
    imagenes = procesador.cargar_imagenes()
    procesador.mostrar_imagenes(imagenes, num_por_clase=3)
