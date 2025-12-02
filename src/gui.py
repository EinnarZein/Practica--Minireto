'''
Aquí va la interfaz gráfica de la aplicación
En este archivo definimos la clase ImageEditorApp que maneja toda la GUI
Permite cargar imágenes, aplicar filtros y mostrar resultados
'''

#Importamos librerías necesarias 📚
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

#Importamos la lógica de procesamiento de imágenes
from src.logic import ImageProcessor

#Esta clase permite crear la aplicación
class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deteccion y Análisis de Fresas- Mini Reto PDI")
        self.root.geometry("1450x950") 

        # ESTILOS
        self.BTN_FONT = ("Segoe UI", 11)
        self.HEADER_FONT = ("Segoe UI", 12, "bold")
        self.BTN_PADY = 8
        
        # VARIABLES
        self.original_img = None
        self.current_img = None
        self.img2 = None
        self.preview_img_data = None
        
        self.kernel_var = tk.IntVar(value=3)
        self.scalar_var = tk.IntVar(value=0)
        self.min_area_var = tk.IntVar(value=2000) 
        self.max_area_var = tk.IntVar(value=150000)
        
        self.create_gui()

    #--- CREACIÓN DE LA INTERFAZ ---
    def create_gui(self):
        # PANEL
        control_canvas = tk.Canvas(self.root, width=420, bg="#e1e1e1")
        control_canvas.pack(side=tk.LEFT, fill=tk.Y)
        control_scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=control_canvas.yview)
        control_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_frame = tk.Frame(control_canvas, width=420, bg="#e1e1e1", padx=20, pady=20)
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        
        # 1. ARCHIVOS
        tk.Label(control_frame, text="1. Gestión de Archivos", font=self.HEADER_FONT, bg="#e1e1e1").pack(anchor="w", pady=(0, 5))
        tk.Button(control_frame, text="📂 Cargar Imagen Principal", font=self.BTN_FONT, command=self.load_main_image, bg="white", pady=self.BTN_PADY).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="📂 Cargar Secundaria (Img 2)", font=self.BTN_FONT, command=self.load_secondary_image, bg="white", pady=self.BTN_PADY).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="↩️ Restaurar Original", font=self.BTN_FONT, bg="#FFCDD2", fg="#B71C1C", command=self.reset_image, pady=self.BTN_PADY).pack(fill=tk.X, pady=5)
        
        tk.Frame(control_frame, height=2, bg="#BDBDBD").pack(fill=tk.X, pady=15)

        # 2. PARÁMETROS
        tk.Label(control_frame, text="2. Parámetros", font=self.HEADER_FONT, bg="#e1e1e1").pack(anchor="w", pady=(0, 5))
        param_frame = tk.LabelFrame(control_frame, text="Valores", bg="#e1e1e1")
        param_frame.pack(fill=tk.X, pady=2)

        tk.Label(param_frame, text="Ruido (%):", bg="#e1e1e1").pack(anchor="w")
        self.scale_noise_amount = tk.Scale(param_frame, from_=0, to=50, orient=tk.HORIZONTAL, bg="#e1e1e1")
        self.scale_noise_amount.set(5)
        self.scale_noise_amount.pack(fill=tk.X)
        
        tk.Label(param_frame, text="Gamma (0.1 - 3.0):", bg="#e1e1e1").pack(anchor="w")
        self.scale_gamma = tk.Scale(param_frame, from_=1, to=30, orient=tk.HORIZONTAL, bg="#e1e1e1")
        self.scale_gamma.set(10)
        self.scale_gamma.pack(fill=tk.X)

        tk.Label(param_frame, text="Kernel (Impar >= 3):", bg="#e1e1e1").pack(anchor="w")
        tk.Spinbox(param_frame, from_=3, to=31, increment=2, state="readonly", font=("Segoe UI", 11), textvariable=self.kernel_var).pack(fill=tk.X, pady=2)
        
        tk.Label(param_frame, text="Valor Escalar:", bg="#e1e1e1").pack(anchor="w")
        tk.Spinbox(param_frame, from_=-255, to=255, increment=1, font=("Segoe UI", 11), textvariable=self.scalar_var).pack(fill=tk.X, pady=2)

        # 3. PROCESAMIENTO
        tk.Label(control_frame, text="3. Filtros y Procesamiento", font=self.HEADER_FONT, bg="#e1e1e1").pack(anchor="w", pady=(15, 5))
        
        tk.Label(control_frame, text="Filtros Espaciales:", bg="#e1e1e1").pack(anchor="w")
        
        #Lista de filtros
        self.combo_filters = ttk.Combobox(control_frame, state="readonly", font=("Segoe UI", 11), values=[
            "Ninguno", "Escala de Grises", 
            "Filtro Promediador", "Filtro Prom. Pesado",
            "Filtro Gaussiano", "Filtro Mediana", 
            "Filtro Moda (Optimizado)", "Filtro Mínimo", "Filtro Máximo",
            "Bordes Canny", "Filtro Sobel", "Filtro Prewitt",
            "Filtro Robert", "Filtro Kirsch", "Filtro Laplaciano (Opencv)",
            "Filtro Laplaciano (8 Vecinos)", "Negativo"
        ])
        self.combo_filters.current(0)
        self.combo_filters.pack(fill=tk.X, pady=4)

        #Lista de operaciones
        tk.Label(control_frame, text="Operaciones:", bg="#e1e1e1").pack(anchor="w")
        self.combo_arithmetic = ttk.Combobox(control_frame, state="readonly", font=("Segoe UI", 11), values=[
            "Ninguno", "Sumar Img", "Restar Img", "Multiplicar Img", "AND Img", "OR Img", "XOR Img",
            "Sumar Escalar", "Restar Escalar", "Multiplicar Escalar", "NOT (Invertir)"])
        self.combo_arithmetic.current(0)
        self.combo_arithmetic.pack(fill=tk.X, pady=4)

        tk.Label(control_frame, text="Brillo / Ecualización:", bg="#e1e1e1").pack(anchor="w")

        #Lista de brillo y ecualización
        self.combo_brightness = ttk.Combobox(control_frame, state="readonly", font=("Segoe UI", 11), values=[
            "Ninguno", "Ecualización Uniforme", "Ecualización Exponencial",
            "Ecualización Rayleigh", "Ecualización Hiperbólica", "Ecualización Hipercúbica",
            "Función Potencia (Gamma)", "Corrección Gamma (Auto)"])
        self.combo_brightness.current(0)
        self.combo_brightness.pack(fill=tk.X, pady=4)

        tk.Label(control_frame, text="Segmentación Clásica:", bg="#e1e1e1").pack(anchor="w")

        # Lista de segmentación clásica
        self.combo_segmentation = ttk.Combobox(control_frame, state="readonly", font=("Segoe UI", 11), values=[
            "Ninguno", "Umbralado Otsu", "Umbralado Kapur (Entropía)", 
            "Mínimo del Histograma", "Multiumbralización (K-Means)"])
        self.combo_segmentation.current(0)
        self.combo_segmentation.pack(fill=tk.X, pady=4)
        
        tk.Label(control_frame, text="Ruido:", bg="#e1e1e1").pack(anchor="w")
        self.combo_noise = ttk.Combobox(control_frame, state="readonly", font=("Segoe UI", 11), values=["Ninguno", "Ruido Gaussiano", "Sal y Pimienta"])
        self.combo_noise.current(0)
        self.combo_noise.pack(fill=tk.X, pady=4)

        # 4. FLUJO
        tk.Frame(control_frame, height=2, bg="#BDBDBD").pack(fill=tk.X, pady=15)
        tk.Button(control_frame, text="👁️ PREVISUALIZAR", font=self.BTN_FONT, bg="#2196F3", fg="white", pady=self.BTN_PADY, command=self.preview_process).pack(fill=tk.X, pady=4)
        tk.Button(control_frame, text="✅ APLICAR", font=self.BTN_FONT, bg="#4CAF50", fg="white", pady=self.BTN_PADY, command=self.apply_change_permanent).pack(fill=tk.X, pady=4)
        
        # 5.Orientado a proyecto final
        tk.Frame(control_frame, height=2, bg="#BDBDBD").pack(fill=tk.X, pady=15)
        tk.Label(control_frame, text="5. Proyecto Fresas", font=self.HEADER_FONT, bg="#e1e1e1", fg="#673AB7").pack(anchor="w")

        tk.Button(control_frame, text="✂️ Eliminar Fondo (HSV)", font=self.BTN_FONT, bg="#7E57C2", fg="white", pady=self.BTN_PADY, 
                  command=self.remove_background_hsv).pack(fill=tk.X, pady=4)
        tk.Button(control_frame, text="🔍 Contar Hojas", font=self.BTN_FONT, bg="#FF9800", fg="white", pady=self.BTN_PADY, 
                  command=self.analyze_objects).pack(fill=tk.X, pady=4)
        tk.Button(control_frame, text="📊 Collage Análisis", font=self.BTN_FONT, bg="#009688", fg="white", pady=self.BTN_PADY, 
                  command=self.show_full_collage).pack(fill=tk.X, pady=4)

        analysis_frame = tk.LabelFrame(control_frame, text="Filtro Tamaño", bg="#e1e1e1")
        analysis_frame.pack(fill=tk.X, pady=5)
        tk.Label(analysis_frame, text="Área Mínima:", bg="#e1e1e1").pack(anchor="w")
        tk.Spinbox(analysis_frame, from_=1, to=10000, increment=50, font=("Segoe UI", 11), textvariable=self.min_area_var).pack(fill=tk.X, pady=2)
        
        self.lbl_count_result = tk.Label(control_frame, text="Objetos: 0", font=("Segoe UI", 14, "bold"), fg="#D32F2F", bg="#e1e1e1")
        self.lbl_count_result.pack(fill=tk.X, pady=10)

        # PANEL DERECHO
        display_frame = tk.Frame(self.root, bg="white")
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.lbl_current = tk.Label(display_frame, text="Original", bg="#f5f5f5", relief="groove")
        self.lbl_current.place(relx=0.02, rely=0.05, relwidth=0.47, relheight=0.6)
        self.lbl_preview = tk.Label(display_frame, text="Resultado", bg="#f5f5f5", relief="groove")
        self.lbl_preview.place(relx=0.51, rely=0.05, relwidth=0.47, relheight=0.6)
        self.lbl_info = tk.Label(display_frame, text="Carga imagen.", bg="white", font=("Segoe UI", 11), justify="left")
        self.lbl_info.place(relx=0.02, rely=0.7, relwidth=0.96)

    # --- FUNCIONES ---

    # CARGA DE IMÁGENES
    def load_main_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        try:
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            if img is None: raise ValueError("Error carga")
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_img = img.copy()
            self.current_img = img.copy()
            self.show_image(self.current_img, self.lbl_current)
            self.lbl_info.config(text=f"Imagen: {img.shape[1]}x{img.shape[0]}")
            self.lbl_count_result.config(text="Objetos: 0")
        except Exception as e: messagebox.showerror("Error", str(e))

    #--- CARGA IMAGEN SECUNDARIA ---
    def load_secondary_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        try:
            stream = open(path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.img2 = img
            messagebox.showinfo("Info", "Imagen 2 cargada.")
        except Exception as e: messagebox.showerror("Error", str(e))

    # MUESTRA IMAGEN EN LABEL
    def show_image(self, img_arr, label_widget):
        if img_arr is None: return
        pil_image = Image.fromarray(img_arr)
        w_box, h_box = label_widget.winfo_width(), label_widget.winfo_height()
        if w_box < 10: w_box, h_box = 400, 300
        pil_image.thumbnail((w_box, h_box))
        tk_image = ImageTk.PhotoImage(pil_image)
        label_widget.config(image=tk_image, text="")
        label_widget.image = tk_image

    # --- LLAMADAS A LÓGICA ---
    def process_algorithm(self, input_img):
        result = input_img.copy()
        
        try:
            k_size = self.kernel_var.get()
            if k_size % 2 == 0: k_size += 1
            noise_pct = self.scale_noise_amount.get() / 100.0
            gamma_val = self.scale_gamma.get() / 10.0
            scalar = self.scalar_var.get()
        except: k_size, noise_pct, gamma_val, scalar = 3, 0.05, 1.0, 0
        
        # 1. Filtros
        filt = self.combo_filters.get()
        if filt != "Ninguno":
            result = ImageProcessor.apply_filters(result, filt, k_size)

        # 2. Operaciones
        op_arit = self.combo_arithmetic.get()
        if "Img" in op_arit and self.img2 is not None:
            img2_r = cv2.resize(self.img2, (result.shape[1], result.shape[0]))
            if "Sumar" in op_arit: result = cv2.addWeighted(result, 0.5, img2_r, 0.5, 0)
            elif "Restar" in op_arit: result = cv2.subtract(result, img2_r)
            elif "Multiplicar" in op_arit: result = cv2.multiply(result, img2_r.astype(np.float32)/255.0).astype(np.uint8)
        elif "Escalar" in op_arit:
            sc_arr = np.full(result.shape, scalar, dtype=result.dtype)
            if "Sumar" in op_arit: result = cv2.add(result, sc_arr)
            elif "Restar" in op_arit: result = cv2.subtract(result, sc_arr)
            elif "Multiplicar" in op_arit: result = cv2.multiply(result, np.clip(sc_arr, 0, 255))
        elif "NOT" in op_arit: result = cv2.bitwise_not(result)
        
        # Ruido (Local o movido a logic)
        noise = self.combo_noise.get()
        if "Sal y Pimienta" in noise:
             # Mini implementación local por rapidez
             for val, color in [(255, 255), (0, 0)]:
                num = int(noise_pct * result.size * 0.5)
                coords = [np.random.randint(0, i - 1, num) for i in result.shape[:2]]
                if len(result.shape) == 3: result[coords[0], coords[1], :] = color
                else: result[coords[0], coords[1]] = color
        elif "Gaussiano" in noise:
             gauss = np.random.normal(0, noise_pct * 255, result.shape)
             result = np.clip(result.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        # 3. Brillo y Segmentación
        bri = self.combo_brightness.get()
        if bri != "Ninguno":
            result = ImageProcessor.apply_brightness(result, bri, gamma_val)
        
        seg = self.combo_segmentation.get()
        if seg != "Ninguno":
            result = ImageProcessor.apply_segmentation(result, seg)
            
        return result

    #--- ELIMINAR FONDO HSV ---
    def remove_background_hsv(self):
        if self.current_img is None: 
            messagebox.showerror("Error", "Carga imagen")
            return
        result = ImageProcessor.remove_background_hsv(self.current_img)
        self.preview_img_data = result
        self.show_image(result, self.lbl_preview)
        if messagebox.askyesno("Confirmar", "¿Usar imagen?"):
            self.current_img = result
            self.show_image(self.current_img, self.lbl_current)

    #--- ANÁLISIS DE OBJETOS ---
    def analyze_objects(self):
        if self.current_img is None: return
        min_area = self.min_area_var.get()
        output_img, count = ImageProcessor.detect_objects(self.current_img, min_area)
        self.lbl_count_result.config(text=f"Objetos: {count}")
        self.show_image(output_img, self.lbl_preview)
        window_name = "Analisis (Redimensionable)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    #-- COLLAGE DE RESULTADOS ---
    def show_full_collage(self):
        if self.current_img is None: return
        k = int(self.kernel_var.get())
        img = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2GRAY) if len(self.current_img.shape) == 3 else self.current_img
        imgs = [
            (img, "Original Gris"),
            (cv2.GaussianBlur(img, (k,k), 0), "Gaussiano"),
            (cv2.blur(img, (k,k)), "Promedio"),
            (cv2.medianBlur(img, k), "Mediana"),
            (cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k), "Sobel"),
            (cv2.Canny(img, 100, 200), "Canny")
        ]
        fig, axs = plt.subplots(2, 3, figsize=(10, 7))
        for i, ax in enumerate(axs.flat):
            if i < len(imgs):
                disp = imgs[i][0]
                if disp.dtype != np.uint8: disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ax.imshow(disp, cmap='gray')
                ax.set_title(imgs[i][1])
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    #--- PREVISUALIZAR Y APLICAR ---
    def preview_process(self):
        if self.current_img is None: return
        self.preview_img_data = self.process_algorithm(self.current_img)
        self.show_image(self.preview_img_data, self.lbl_preview)

    #--- APLICAR CAMBIO PERMANENTE ---
    def apply_change_permanent(self):
        if self.preview_img_data is None: 
            messagebox.showwarning("Atención", "Previsualiza primero.")
            return
        self.current_img = self.preview_img_data
        self.show_image(self.current_img, self.lbl_current)
        self.lbl_preview.config(image="", text="Aplicado")
        self.preview_img_data = None

        self.combo_filters.current(0)
        self.combo_arithmetic.current(0)
        self.combo_brightness.current(0)
        self.combo_segmentation.current(0)

    #--- RESTAURAR IMAGEN ORIGINAL ---
    def reset_image(self):
        if self.original_img is None: return
        self.current_img = self.original_img.copy()
        self.show_image(self.current_img, self.lbl_current)
        self.lbl_preview.config(image="", text="Reset")
        self.lbl_count_result.config(text="Objetos: 0")