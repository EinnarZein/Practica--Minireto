import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# FUNCIONES DE RUIDO
# ----------------------------------

# Ruido sal y pimienta
def agregar_ruido_sal_pimienta(imagen, cantidad=0.02):
    imagen_ruido = imagen.copy()
    filas, columnas = imagen.shape
    num_ruido = int(cantidad * filas * columnas)
    for _ in range(num_ruido):
        i = np.random.randint(0, columnas)
        j = np.random.randint(0, filas)
        if np.random.rand() < 0.5:
            imagen_ruido[j, i] = 0     # Pimienta (negro)
        else:
            imagen_ruido[j, i] = 255   # Sal (blanco)
    return imagen_ruido

# Ruido gaussiano
def agregar_ruido_gaussiano(imagen, media=0, sigma=20):
    gauss = np.random.normal(media, sigma, imagen.shape).astype(np.int16)
    imagen_ruido = imagen.astype(np.int16) + gauss
    imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)
    return imagen_ruido

# ----------------------------------
# FILTROS
# ----------------------------------

# Filtro promediador simple
def filtro_promediador(imagen, ksize=3):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(imagen, -1, kernel)

# Filtro promediador pesado (más peso al centro)
def filtro_promediador_pesado(imagen):
    kernel = np.array([[1, 1, 1],
                       [1, 2, 1],
                       [1, 1, 1]], dtype=np.float32)
    kernel = kernel / kernel.sum()
    return cv2.filter2D(imagen, -1, kernel)

# Filtro gaussiano
def filtro_gaussiano(imagen, ksize=5, sigma=1):
    return cv2.GaussianBlur(imagen, (ksize, ksize), sigma)

# Filtro de mediana
def filtro_mediana(imagen, ksize=3):
    return cv2.medianBlur(imagen, ksize)

# Filtro de moda (ventana ksize x ksize)
def filtro_moda(imagen, ksize=3):
    pad = ksize // 2
    # Borde reflejado para evitar problemas en los bordes
    padded = cv2.copyMakeBorder(imagen, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    salida = np.zeros_like(imagen)

    filas, columnas = imagen.shape
    for i in range(filas):
        for j in range(columnas):
            ventana = padded[i:i+ksize, j:j+ksize].flatten()
            conteo = np.bincount(ventana)
            salida[i, j] = np.argmax(conteo)
    return salida

# Filtro máximo
def filtro_maximo(imagen, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(imagen, kernel)

# Filtro mínimo
def filtro_minimo(imagen, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(imagen, kernel)

# ----------------------------------
# FUNCIONES DE APOYO PARA MOSTRAR
# ----------------------------------

def mostrar_dos(im1, titulo1, im2, titulo2):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(im1, cmap='gray')
    plt.title(titulo1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(im2, cmap='gray')
    plt.title(titulo2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------------
# PROGRAMA PRINCIPAL CON MENÚ
# ----------------------------------

def main():
    # Variables globales de estado
    imagen_color = None
    imagen_gris = None

    imagen_ruido_gaussiano = None
    imagen_ruido_sal_pimienta = None

    img_prom = None
    img_prom_pesado = None
    img_gauss_filtro = None
    img_mediana = None
    img_moda = None
    img_max = None
    img_min = None

    while True:
        print("\n===== MENÚ DE PROCESAMIENTO DE IMÁGENES =====")
        print("1. Cargar imagen y convertir a escala de grises")
        print("2. Generar ruido gaussiano")
        print("3. Generar ruido sal y pimienta")
        print("4. Filtro promediador (antes y después)")
        print("5. Filtro promediador pesado (antes y después)")
        print("6. Filtro Gaussiano (antes y después sobre ruido gaussiano)")
        print("7. Filtro de mediana (antes y después)")
        print("8. Filtro Moda (antes y después)")
        print("9. Filtro máximo/mínimo (antes y después)")
        print("10. Mostrar collage completo")
        print("11. Cerrar")
        opcion = input("Selecciona una opción: ")

        # ------------------------------------------
        # 1. Cargar imagen y convertir a escala de grises
        # ------------------------------------------
        if opcion == '1':
            # Puedes cambiar '2.jpg' por el nombre de tu archivo
            ruta = '1.jpg'
            imagen_color = cv2.imread(ruta)
            if imagen_color is None:
                print("Error: no se pudo cargar la imagen. Verifica la ruta/nombre.")
                continue
            imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
            mostrar_dos(imagen_color[:, :, ::-1], 'Imagen original (color)', imagen_gris, 'Imagen en escala de grises')

        # ------------------------------------------
        # 2. Generar ruido gaussiano
        # ------------------------------------------
        elif opcion == '2':
            if imagen_gris is None:
                print("Primero carga la imagen (opción 1).")
                continue
            imagen_ruido_gaussiano = agregar_ruido_gaussiano(imagen_gris, media=0, sigma=20)
            mostrar_dos(imagen_gris, 'Imagen en gris', imagen_ruido_gaussiano, 'Imagen con ruido gaussiano')

        # ------------------------------------------
        # 3. Generar ruido sal y pimienta
        # ------------------------------------------
        elif opcion == '3':
            if imagen_gris is None:
                print("Primero carga la imagen (opción 1).")
                continue
            imagen_ruido_sal_pimienta = agregar_ruido_sal_pimienta(imagen_gris, cantidad=0.02)
            mostrar_dos(imagen_gris, 'Imagen en gris', imagen_ruido_sal_pimienta, 'Imagen con ruido sal y pimienta')

        # ------------------------------------------
        # 4. Filtro promediador
        # ------------------------------------------
        elif opcion == '4':
            # Usamos la imagen con ruido sal y pimienta por defecto
            if imagen_ruido_sal_pimienta is None:
                print("Primero genera ruido sal y pimienta (opción 3).")
                continue
            img_prom = filtro_promediador(imagen_ruido_sal_pimienta, ksize=3)
            mostrar_dos(imagen_ruido_sal_pimienta, 'Antes (ruido sal y pimienta)', img_prom, 'Después (filtro promediador)')

        # ------------------------------------------
        # 5. Filtro promediador pesado
        # ------------------------------------------
        elif opcion == '5':
            if imagen_ruido_sal_pimienta is None:
                print("Primero genera ruido sal y pimienta (opción 3).")
                continue
            img_prom_pesado = filtro_promediador_pesado(imagen_ruido_sal_pimienta)
            mostrar_dos(imagen_ruido_sal_pimienta, 'Antes (ruido sal y pimienta)', img_prom_pesado, 'Después (promediador pesado)')

        # ------------------------------------------
        # 6. Filtro Gaussiano sobre ruido gaussiano
        # ------------------------------------------
        elif opcion == '6':
            if imagen_ruido_gaussiano is None:
                print("Primero genera ruido gaussiano (opción 2).")
                continue
            img_gauss_filtro = filtro_gaussiano(imagen_ruido_gaussiano, ksize=5, sigma=1)
            mostrar_dos(imagen_ruido_gaussiano, 'Antes (ruido gaussiano)', img_gauss_filtro, 'Después (filtro gaussiano)')

        # ------------------------------------------
        # 7. Filtro de mediana
        # ------------------------------------------
        elif opcion == '7':
            # Lo aplicamos a la imagen con ruido sal y pimienta (es donde más se nota)
            if imagen_ruido_sal_pimienta is None:
                print("Primero genera ruido sal y pimienta (opción 3).")
                continue
            img_mediana = filtro_mediana(imagen_ruido_sal_pimienta, ksize=3)
            mostrar_dos(imagen_ruido_sal_pimienta, 'Antes (ruido sal y pimienta)', img_mediana, 'Después (filtro mediana)')

        # ------------------------------------------
        # 8. Filtro Moda
        # ------------------------------------------
        elif opcion == '8':
            if imagen_ruido_sal_pimienta is None:
                print("Primero genera ruido sal y pimienta (opción 3).")
                continue
            print("Calculando filtro de moda (puede tardar un poco)...")
            img_moda = filtro_moda(imagen_ruido_sal_pimienta, ksize=3)
            mostrar_dos(imagen_ruido_sal_pimienta, 'Antes (ruido sal y pimienta)', img_moda, 'Después (filtro moda)')

        # ------------------------------------------
        # 9. Filtro máximo / mínimo
        # ------------------------------------------
        elif opcion == '9':
            # También los aplicamos sobre la imagen con ruido sal y pimienta
            if imagen_ruido_sal_pimienta is None:
                print("Primero genera ruido sal y pimienta (opción 3).")
                continue
            img_max = filtro_maximo(imagen_ruido_sal_pimienta, ksize=3)
            img_min = filtro_minimo(imagen_ruido_sal_pimienta, ksize=3)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(imagen_ruido_sal_pimienta, cmap='gray')
            plt.title('Antes (ruido sal y pimienta)')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(img_max, cmap='gray')
            plt.title('Después (filtro máximo)')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(img_min, cmap='gray')
            plt.title('Después (filtro mínimo)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        # ------------------------------------------
        # 10. Collage
        # ------------------------------------------
        elif opcion == '10':
            if imagen_gris is None or imagen_color is None:
                print("Primero carga la imagen (opción 1).")
                continue

            # Lista de imágenes y títulos (solo se mostrarán las que existan)
            imagenes = []
            titulos = []

            imagenes.append(imagen_color[:, :, ::-1])  # BGR -> RGB
            titulos.append('Original (color)')

            imagenes.append(imagen_gris)
            titulos.append('Escala de grises')

            if imagen_ruido_gaussiano is not None:
                imagenes.append(imagen_ruido_gaussiano)
                titulos.append('Ruido gaussiano')

            if imagen_ruido_sal_pimienta is not None:
                imagenes.append(imagen_ruido_sal_pimienta)
                titulos.append('Ruido sal y pimienta')

            if img_prom is not None:
                imagenes.append(img_prom)
                titulos.append('Filtro promediador')

            if img_prom_pesado is not None:
                imagenes.append(img_prom_pesado)
                titulos.append('Promediador pesado')

            if img_gauss_filtro is not None:
                imagenes.append(img_gauss_filtro)
                titulos.append('Filtro gaussiano')

            if img_mediana is not None:
                imagenes.append(img_mediana)
                titulos.append('Filtro mediana')

            if img_moda is not None:
                imagenes.append(img_moda)
                titulos.append('Filtro moda')

            if img_max is not None:
                imagenes.append(img_max)
                titulos.append('Filtro máximo')

            if img_min is not None:
                imagenes.append(img_min)
                titulos.append('Filtro mínimo')

            n = len(imagenes)
            if n == 0:
                print("No hay imágenes suficientes para el collage.")
                continue

            # Definimos tamaño aproximado de rejilla
            columnas = 4
            filas = int(np.ceil(n / columnas))

            plt.figure(figsize=(4 * columnas, 3 * filas))
            for idx, (im, titulo) in enumerate(zip(imagenes, titulos), start=1):
                plt.subplot(filas, columnas, idx)
                if im.ndim == 2:
                    plt.imshow(im, cmap='gray')
                else:
                    plt.imshow(im)
                plt.title(titulo)
                plt.axis('off')

            plt.tight_layout()
            plt.show()

        # ------------------------------------------
        # 11. Cerrar
        # ------------------------------------------
        elif opcion == '11':
            print("Cerrando programa...")
            break

        else:
            print("Opción no válida, intenta de nuevo.")

if __name__ == '__main__':
    main()
