import cv2
import numpy as np
import os
#Iker Montoya 0105

# Cargar imagen desde la carpeta imagenes
ruta = os.path.join(os.path.dirname(__file__), "..", "imagenes", "pinguino.jpg")
imagen = cv2.imread(ruta)

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    print("Ruta buscada:", ruta)
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(gris_float, 2, 3, 0.04)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(esquinas, None)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.01 * esquinas.max()

# Marcar esquinas en rojo
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar imagen original
cv2.imshow("Pinguino original", imagen)

# Mostrar esquinas detectadas
cv2.imshow("Esquinas detectadas", resultado)

# Guardar resultado en resultados
ruta_resultado = os.path.join(
    os.path.dirname(__file__),
    "..",
    "resultados",
    "pinguino_esquinas.jpg"
)

cv2.imwrite(ruta_resultado, resultado)

# Contar esquinas
cantidad_esquinas = np.sum(esquinas > umbral)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:", cantidad_esquinas)
print("Resultado guardado en:", ruta_resultado)

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
print("Iker Montoya 0105")