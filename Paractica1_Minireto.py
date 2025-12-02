import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk  # Importamos tkinter completo
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- 1. DEFINICIÓN DE FUNCIONES AUXILIARES ---

def show_resized(window_name, image, scale=0.7):
    
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    width = max(1, width)
    height = max(1, height)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, resized_image)

def binarizar(imagen):
    """
    Convierte una imagen BGR a escala de grises y luego a binaria
    usando un umbral fijo de 120.
    """
    gris_local = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris_local, 120, 255, cv2.THRESH_BINARY)
    return binaria

def etiquetar_componentes(img_binaria, nombre, conectividad):
    """
    Aplica y MUESTRA el etiquetado de componentes conexas.
    Esta función ahora es "modal": muestra su ventana y espera.
    """
    num_labels, etiquetas, stats, centroids = cv2.connectedComponentsWithStats(
        img_binaria,
        connectivity=conectividad
    )
    img_color = cv2.applyColorMap((etiquetas * 15).astype(np.uint8), cv2.COLORMAP_JET)
    componentes = num_labels - 1
    print(f"[{nombre}] Conectividad {conectividad}: {componentes} componentes detectadas.")
    
    window_title = f"{nombre} - Conectividad {conectividad} ({componentes} componentes)"
    show_resized(window_title, img_color)
    cv2.waitKey(0) # Espera a que el usuario presione una tecla
    cv2.destroyWindow(window_title) # Cierra solo esta ventana

# --- 2. CARGA Y PRE-CÁLCULO DE IMÁGENES ---
# Esto se ejecuta una sola vez al inicio del script.

Tk().withdraw()  # Oculta la ventana principal de Tkinter

print("=== Práctica 1: Procesamiento básico de imágenes ===\n")
print("Cargando imágenes... Por favor, selecciona los 3 archivos.")

ruta_img1 = askopenfilename(title="Selecciona IM1", filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
img1 = cv2.imread(ruta_img1)

ruta_img2 = askopenfilename(title="Selecciona IM2", filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
img2 = cv2.imread(ruta_img2)

ruta_binaria = askopenfilename(title="Selecciona imagen binaria para Etiquetado", filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
image = cv2.imread(ruta_binaria, cv2.IMREAD_GRAYSCALE)

# Validación y preparación
if img1 is None or img2 is None or image is None:
    print("\n[ERROR] Una o más imágenes no se cargaron. Saliendo del programa.")
    exit()

print("\nImágenes cargadas y procesando... Por favor, espera.")

# Ajuste de tamaño
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Pre-cálculo Sección 1: Originales
originales = np.hstack((img1, img2))

# Pre-cálculo Sección 2: Aritméticas
suma = cv2.add(img1, img2)
resta = cv2.subtract(img1, img2)
multiplicacion = cv2.multiply(img1, 1.2)
aritmeticas = np.hstack((suma, resta, multiplicacion))

# Pre-cálculo Sección 3: Lógicas
and_logico = cv2.bitwise_and(img1, img2)
or_logico = cv2.bitwise_or(img1, img2)
xor_logico = cv2.bitwise_xor(img1, img2)
not_logico = cv2.bitwise_not(img1)
fila1 = np.hstack((and_logico, or_logico))
fila2 = np.hstack((xor_logico, not_logico))
logicas = np.vstack((fila1, fila2))

# Pre-cálculo Sección 4: Grises y Brillo
gris = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
brillo_ajustado = cv2.convertScaleAbs(gris, alpha=1.1, beta=25)

# Pre-cálculo Sección 5: Umbralización
umbrales = []
for t in [80, 135, 145]:
    _, umbral = cv2.threshold(brillo_ajustado, t, 255, cv2.THRESH_BINARY)
    umbrales.append(umbral)
fila_umbral = np.hstack((brillo_ajustado, umbrales[0], umbrales[1], umbrales[2]))

# Pre-cálculo Sección 6: Datos para etiquetado
bin_original = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY)[1]
bin_suma = binarizar(suma)
bin_resta = binarizar(resta)
bin_and = binarizar(and_logico) # Asumiendo que binarizar maneja 3 canales
bin_or = binarizar(or_logico)
bin_xor = binarizar(xor_logico)
bin_not = binarizar(not_logico)
bin_umbral = cv2.threshold(brillo_ajustado, 120, 255, cv2.THRESH_BINARY)[1]

imagenes_y_nombres = [
    (bin_original, "IM1 binarizada"), (bin_suma, "Suma (IM1 + IM2)"),
    (bin_resta, "Resta (IM1 - IM2)"), (bin_and, "Operación lógica AND"),
    (bin_or, "Operación lógica OR"), (bin_xor, "Operación lógica XOR"),
    (bin_not, "Operación lógica NOT"), (bin_umbral, "Brillo ajustado t=120")
]

# Pre-cálculo Sección 7: Datos para conteo
_, binary_image_conteo = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours_conteo, _ = cv2.findContours(binary_image_conteo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total_objetos_conteo = len(contours_conteo)

print("¡Procesamiento completado! Mostrando menú principal.")

# --- 3. DEFINICIÓN DE FUNCIONES DEL MENÚ ---
# Estas funciones se llaman desde los botones de Tkinter.

def f_show_originales():
    window_name = "Imágenes originales: IM1 | IM2"
    show_resized(window_name, originales)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def f_show_aritmeticas():
    window_name = "Operaciones aritméticas: Suma | Resta | Multiplicación"
    show_resized(window_name, aritmeticas, scale=0.4) # Más pequeña
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def f_show_logicas():
    window_name = "Operaciones lógicas: AND | OR | XOR | NOT (IM1)"
    show_resized(window_name, logicas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def f_show_histograma():
    print("Mostrando Histograma (en ventana de Matplotlib)...")
    plt.figure(figsize=(8, 4))
    plt.title("Histograma de niveles de gris - IM1")
    plt.xlabel("Nivel de gris")
    plt.ylabel("Frecuencia")
    plt.hist(gris.ravel(), bins=256, range=[0, 256])
    plt.show() # Esta es una ventana bloqueante. Al cerrarla, vuelves al menú.

def f_show_brillo():
    window_name = "IM1 en escala de grises con brillo ajustado"
    show_resized(window_name, brillo_ajustado)
    cv2.waitKey(0)
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        # Si falla (porque ya se cerró con 'X'), simplemente continúa
        pass

def f_show_umbrales():
    window_name = "Brillo ajustado | Umbral t=80 | t=135 | t=145"
    show_resized(window_name, fila_umbral, scale=0.4) # Más pequeña
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def f_run_etiquetado():
    print("\nIniciando secuencia de etiquetado (Sección 6)...")
    print("Presiona una tecla en cada imagen para continuar.")
    
    # Este bucle ahora es a prueba de errores
    for img_bin, nombre in imagenes_y_nombres:
        try:
            # Intenta procesar y mostrar conectividad 4
            etiquetar_componentes(img_bin, nombre, 4)
            
            # Intenta procesar y mostrar conectividad 8
            etiquetar_componentes(img_bin, nombre, 8)
            
        except Exception as e:
            # Si algo falla (ej. error de formato de imagen, 
            # o el error de cerrar la ventana con 'X'),
            # imprime el error y continúa con la siguiente imagen.
            print(f"--- ERROR al procesar '{nombre}': {e}")
            print("--- Saltando a la siguiente imagen de la secuencia...")
            continue # Continúa con la siguiente iteración del bucle 'for'
            
    print("Secuencia de etiquetado finalizada.")

def f_run_conteo_detallado():
    print("\nMostrando análisis de conteo detallado (Sección 7)...")
    
    # --- Bloque 1: Mostrar imagen binaria base ---
    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image_conteo, cmap='gray')
        plt.title('Imagen binarizada para análisis de objetos')
        plt.axis('off')
        plt.show() # Esta es una ventana bloqueante
    except Exception as e:
        print(f"--- ERROR al mostrar la imagen binaria base: {e}")

    # --- Bloque 2: Preparar y mostrar conteo detallado ---
    try:
        # Preparar imagen de conteo
        image_color_conteo = cv2.cvtColor(binary_image_conteo, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours_conteo):
            cv2.drawContours(image_color_conteo, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image_color_conteo, f'{i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar imagen de conteo
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image_color_conteo, cv2.COLOR_BGR2RGB))
        plt.title(f'Objetos detectados y numerados\nTotal: {total_objetos_conteo}')
        plt.axis('off')
        plt.show() # Esta es otra ventana bloqueante
    
    except Exception as e:
        print(f"--- ERROR al mostrar la imagen de conteo detallado: {e}")

    print("Análisis de conteo finalizado.")

# --- 4. CREACIÓN Y EJECUCIÓN DEL MENÚ PRINCIPAL ---

def exit_program():
    print("\nSaliendo del programa...")
    try:
        plt.close('all')  # Cierra todas las figuras de Matplotlib
    except Exception as e:
        print(f"Error al cerrar Matplotlib: {e}")
    
    try:
        root.quit()       # Detiene el mainloop de Tkinter
    except Exception as e:
        print(f"Error al hacer quit() de Tkinter: {e}")
    
    try:
        root.destroy()    # Destruye la ventana de Tkinter
    except Exception as e:
        print(f"Error al hacer destroy() de Tkinter: {e}")



root = tk.Tk()
root.title("Menú Principal - Práctica 1 PDI")
root.geometry("400x450") # Tamaño de la ventana del menú

# Crear un marco para los botones
frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True)

# Estilo de los botones
btn_style = {'pady': 5, 'fill': 'x', 'expand': True}

tk.Label(frame, text="Selecciona una operación:", font=("Helvetica", 14, "bold")).pack(pady=(0, 15))

tk.Button(frame, text="1. Ver Imágenes Originales", command=f_show_originales).pack(**btn_style)
tk.Button(frame, text="2. Operaciones Aritméticas", command=f_show_aritmeticas).pack(**btn_style)
tk.Button(frame, text="3. Operaciones Lógicas", command=f_show_logicas).pack(**btn_style)
tk.Button(frame, text="4. Histograma (IM1)", command=f_show_histograma).pack(**btn_style)
tk.Button(frame, text="5. Ajuste de Brillo (IM1)", command=f_show_brillo).pack(**btn_style)
tk.Button(frame, text="6. Umbralización Múltiple", command=f_show_umbrales).pack(**btn_style)
tk.Button(frame, text="7. Etiquetado Múltiple (Secuencia)", command=f_run_etiquetado).pack(**btn_style)
tk.Button(frame, text="8. Conteo Detallado de Objetos", command=f_run_conteo_detallado).pack(**btn_style)

tk.Button(frame, text="Salir", command=exit_program, bg="#531919", fg="white", font=("Helvetica", 10, "bold")).pack(pady=(20, 0), fill='x')

# Iniciar el bucle de la aplicación
root.mainloop()

# Limpieza final al cerrar el menú
cv2.destroyAllWindows()
print("\n=== Programa finalizado. ===")