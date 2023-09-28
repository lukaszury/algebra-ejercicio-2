import numpy as np
import numpy.linalg as la
from skimage.io import imread, imshow
import cv2
import matplotlib.pyplot as plt

#Función para recortar imagenes
def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    """
    Esta función recibe una imagen y devuelve otra imagen recortada.

    Args:
      ruta_img (str): Ruta de la imagen original que se desea recortar.
      ruta_img_crop (str): Ruta donde se guardará la imagen recortada.
      x_inicial (int): Coordenada x inicial del área de recorte.
      x_final (int): Coordenada x final del área de recorte.
      y_inicial (int): Coordenada y inicial del área de recorte.
      y_final (int): Coordenada y final del área de recorte.

    Return
      None
    """
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))

#Ruta de imagenes e imagenes recortadas
ruta_imagen_1 = "imagen1.jpg"
ruta_imagen_2 = "imagen2.png"


# 1 -Cargar dos imagenes a elección utilizando imread y mostrarlas como imagen.
imagen_1 = imread(ruta_imagen_1)
imagen_2 = imread(ruta_imagen_2)

plt.imshow(imagen_1)
plt.title("Imagen 1")
plt.show()

plt.imshow(imagen_2)
plt.title("Imagen 2")
plt.show()

# 2 - Imprimir el tamaño de cada una de las imágenes.

print(f"Tamaño de la imagen 1: {imagen_1.shape}")
print(f"Tamaño de la imagen 2: {imagen_2.shape}")

# 3 - Recortar ambas imágenes para que tengan el mismo tamaño, siendo un requisito que la imagen sea cuadrada. Mostrar el resultado obtenido como imágenes.

ruta_imagen_3 = "imagen2.jpg"
ruta_imagen_4 = "imagen2.png"
imagen_3 = imread(ruta_imagen_3)


recortar_imagen_v2(ruta_imagen_1,ruta_imagen_3,0,400,200,600)

# Si queremos hacer operaciones, como sumarlas, restarlas, o cualquier otra operación con matrices o en este caso con 
# imagenes, es esencial que ambas imagenes tengan el mismo tamaño

# 4 - Para una de las imágenes recortadas, mostrarla como una matriz. Imprimir el tamaño.

# Matriz
print(imagen_3)

# Imagen
plt.imshow(imagen_3)
plt.title("Imagen 1 recortada")
plt.show()

print(f"Tamaño de la imagen recortada: {imagen_3.shape}")


# 5 - Calcular la matriz traspuesta de las imágenes del punto 3. Mostrarlas como matriz y como imagen. Comentar los resultados.

imagen1_traspuesta = np.transpose(imagen_3, (1, 0, 2))

# Matriz
print(imagen1_traspuesta)

# Imagen
plt.imshow(imagen1_traspuesta)
plt.title("Imagen 3 Traspuesta")
plt.show()

# 6 - Convertir y mostrar las imagenes recortadas a escala de grises.

# Calculo la media sobre el array de los canales RGB para obtener el promedio
imagen1_grayscale = np.mean(imagen_3, axis=-1)

plt.imshow(imagen1_grayscale, cmap='gray')
plt.title("Imagen 1 Recortada en Escala de Grises")
plt.show()

# 7 - Verificar para cada una de las matrices correspondientes a las imágenes recortadas, si existe su inversa y en caso de que exista, calcular.

#Normalizo la imagen para asegurarme de que va a estar dentro de los valor para que no se desborde
imagen1_normalized = imagen1_grayscale / 255.0

#Calculo la determinante
det1 = la.det(imagen1_normalized)

#Determino si existe inversa
if det1 == 0:
    print("La matriz de la imagen no tiene inversa.")
else:
    inversa_imagen1 = la.inv(imagen1_grayscale)
    print("La matriz de la imagen tiene inversa.")

# 8 - Producto de una matriz por un escalar
escalar1 = 1.5
escalar2 = 0.5

#Multiplico imagenes por escalar
imagen3_escalar1 = imagen1_grayscale * escalar1
imagen3_escalar2 = imagen1_grayscale * escalar2

imagen1_contraste1 = np.clip(imagen3_escalar1, 0, 255)
imagen1_contraste2 = np.clip(imagen3_escalar2, 0, 255)

plt.imshow(imagen1_contraste1, cmap='gray')
plt.title("Imagen 1 con α > 1")
plt.show()

plt.imshow(imagen1_contraste2, cmap='gray')
plt.title("Imagen 1 con 0 < α < 1")
plt.show()

# 9 - Multipicación de matrices y prueba de que la multiplicación de matrices no es conmutativa:

tamanio = imagen1_grayscale.shape[0]
identidad = np.eye(tamanio)
W = np.fliplr(identidad)

imagen_volteada = W @ imagen1_grayscale
imagen_volteada_inversa = imagen1_grayscale @ W

plt.imshow(imagen_volteada, cmap='gray')
plt.title("Imagen volteada con W @ imagen")
plt.show()

plt.imshow(imagen_volteada_inversa, cmap='gray')
plt.title("Imagen volteada con imagen @ W")
plt.show()

# 10 - Calcular el negativo de una de las imagenes utilizando la resta de matrices.

matriz_auxiliar = np.full(imagen1_grayscale.shape, 255)
imagen_negativa = matriz_auxiliar - imagen1_grayscale

plt.imshow(imagen_negativa, cmap='gray')
plt.title("Negativo de la Imagen")
plt.show()



