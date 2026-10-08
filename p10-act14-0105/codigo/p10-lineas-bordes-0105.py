import cv2
import numpy as np
import os

# Iker Montoya 0105

# Cargar imagen
ruta = os.path.join(os.path.dirname(__file__), "..", "imagenes", "pinguino.jpg")
imagen = cv2.imread(ruta)

# Verificar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    print("Ruta:", ruta)
    exit()

# Escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Float32
gris_float = np.float32(gris)

# Detectar esquinas
esquinas = cv2.cornerHarris(gris_float, 2, 3, 0.04)

# Dilatar
esquinas = cv2.dilate(esquinas, None)

# PRUEBA 1
resultado1 = imagen.copy()
umbral1 = 0.01 * esquinas.max()
resultado1[esquinas > umbral1] = [0, 0, 255]

# PRUEBA 2
resultado2 = imagen.copy()
umbral2 = 0.02 * esquinas.max()
resultado2[esquinas > umbral2] = [0, 0, 255]

# PRUEBA 3
resultado3 = imagen.copy()
umbral3 = 0.05 * esquinas.max()
resultado3[esquinas > umbral3] = [0, 0, 255]

# Mostrar resultados
cv2.imshow("Prueba 0.01", resultado1)
cv2.imshow("Prueba 0.02", resultado2)
cv2.imshow("Prueba 0.05", resultado3)

# Guardar resultados
cv2.imwrite("../resultados/pinguino_001.jpg", resultado1)
cv2.imwrite("../resultados/pinguino_002.jpg", resultado2)
cv2.imwrite("../resultados/pinguino_005.jpg", resultado3)

# Mostrar cantidades
print("Iker Montoya 0105")
print("Prueba 0.01:", np.sum(esquinas > umbral1), "esquinas")
print("Prueba 0.02:", np.sum(esquinas > umbral2), "esquinas")
print("Prueba 0.05:", np.sum(esquinas > umbral3), "esquinas")

# Esperar
cv2.waitKey(0)

# Cerrar
cv2.destroyAllWindows()