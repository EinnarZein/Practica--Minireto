
'''
Docstring for src.gui
Módulo de la interfaz gráfica de usuario (GUI) para la aplicación Mini Reto PDI.
'''
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Importación Relativa
try:
    from .logic import ImageProcessor
except ImportError:
    from logic import ImageProcessor

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1450x950") 

        # ESTILOS
        self.BTN_FONT = ("Segoe UI", 10)
        self.HEADER_FONT = ("Segoe UI", 11, "bold")
        
        # VARIABLES
        self.original_img = None
        self.current_img = None
        self.img2 = None
        self.preview_img_data = None
        
        self.kernel_var = tk.IntVar(value=3)
        self.scalar_var = tk.IntVar(value=0)
        self.min_area_var = tk.IntVar(value=200)
        self.noise_var = tk.IntVar(value=5) 
        
        self.create_gui()

    def create_gui(self):
        # PANEL IZQUIERDO
        control_canvas = tk.Canvas(self.root, width=400, bg="#f0f0f0")
        control_canvas.pack(side=tk.LEFT, fill=tk.Y)
        scroll = ttk.Scrollbar(self.root, orient="vertical", command=control_canvas.yview)
        scroll.pack(side=tk.LEFT, fill="y")
        control_canvas.configure(yscrollcommand=scroll.set)
        
        control_frame = tk.Frame(control_canvas, width=400, bg="#f0f0f0", padx=15, pady=15)
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        
        # 1. GESTIÓN DE ARCHIVOS
        tk.Label(control_frame, text="1. Gestión de Archivos", font=self.HEADER_FONT, bg="#f0f0f0").pack(anchor="w")
        tk.Button(control_frame, text="📂 Cargar Imagen Principal", font=self.BTN_FONT, command=self.load_main_image, bg="white").pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="📂 Cargar Secundaria (Op)", font=self.BTN_FONT, command=self.load_secondary_image, bg="white").pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="↩️ Restaurar Original", font=self.BTN_FONT, bg="#FFCDD2", command=self.reset_image).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 2. PARÁMETROS
        tk.Label(control_frame, text="2. Parámetros", font=self.HEADER_FONT, bg="#f0f0f0").pack(anchor="w")
        
        tk.Label(control_frame, text="Kernel (Impar):", bg="#f0f0f0").pack(anchor="w")
        tk.Spinbox(control_frame, from_=3, to=31, increment=2, textvariable=self.kernel_var).pack(fill=tk.X)
        
        tk.Label(control_frame, text="Escalar / Gamma:", bg="#f0f0f0").pack(anchor="w")
        tk.Spinbox(control_frame, from_=-100, to=100, textvariable=self.scalar_var).pack(fill=tk.X)
        self.scale_gamma = tk.Scale(control_frame, from_=1, to=30, orient=tk.HORIZONTAL, bg="#f0f0f0")
        self.scale_gamma.set(10)
        self.scale_gamma.pack(fill=tk.X)
        
        tk.Label(control_frame, text="Ruido (%):", bg="#f0f0f0").pack(anchor="w")
        tk.Scale(control_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.noise_var, bg="#f0f0f0").pack(fill=tk.X)

        # 3. FILTROS Y OPERACIONES
        tk.Label(control_frame, text="3. Procesamiento", font=self.HEADER_FONT, bg="#f0f0f0").pack(anchor="w", pady=(10, 0))
        
        self.combo_filters = ttk.Combobox(control_frame, state="readonly", values=[
            "Ninguno", "Escala de Grises", "Filtro Promediador", "Filtro Prom. Pesado",
            "Filtro Gaussiano", "Filtro Mediana", "Filtro Moda", "Filtro Mínimo", "Filtro Máximo",
            "Bordes Canny", "Filtro Sobel", "Filtro Prewitt", "Filtro Robert", "Filtro Kirsch",
            "Filtro Laplaciano (Opencv)", "Filtro Laplaciano (8 Vecinos)", "Negativo"
        ])
        self.combo_filters.current(0)
        self.combo_filters.pack(fill=tk.X, pady=2)

        tk.Label(control_frame, text="Operaciones:", bg="#f0f0f0").pack(anchor="w")
        self.combo_arithmetic = ttk.Combobox(control_frame, state="readonly", values=[
            "Ninguno", "Sumar Img2", "Restar Img2", "Multiplicar Img2", 
            "AND Img2", "OR Img2", "XOR Img2",
            "Sumar Escalar", "Restar Escalar", "Multiplicar Escalar", "NOT (Invertir)"
        ])
        self.combo_arithmetic.current(0)
        self.combo_arithmetic.pack(fill=tk.X, pady=2)

        tk.Label(control_frame, text="Brillo/Ecualización:", bg="#f0f0f0").pack(anchor="w")
        self.combo_brightness = ttk.Combobox(control_frame, state="readonly", values=[
            "Ninguno", "Ecualización Uniforme", "Ecualización Exponencial",
            "Ecualización Rayleigh", "Ecualización Hiperbólica", "Ecualización Hipercúbica",
            "Función Potencia (Gamma)", "Corrección Gamma (Auto)"
        ])
        self.combo_brightness.current(0)
        self.combo_brightness.pack(fill=tk.X, pady=2)

        tk.Label(control_frame, text="Segmentación:", bg="#f0f0f0").pack(anchor="w")
        self.combo_segmentation = ttk.Combobox(control_frame, state="readonly", values=[
            "Ninguno", "Umbralado Otsu", "Umbralado Kapur (Entropía)", 
            "Mínimo del Histograma", "Multiumbralización (K-Means)"
        ])
        self.combo_segmentation.current(0)
        self.combo_segmentation.pack(fill=tk.X, pady=2)

        tk.Label(control_frame, text="Agregar Ruido:", bg="#f0f0f0").pack(anchor="w")
        self.combo_noise = ttk.Combobox(control_frame, state="readonly", values=["Ninguno", "Ruido Gaussiano", "Sal y Pimienta"])
        self.combo_noise.current(0)
        self.combo_noise.pack(fill=tk.X, pady=2)

        tk.Button(control_frame, text="👁️ Previsualizar", font=self.BTN_FONT, bg="#BBDEFB", command=self.preview_process).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="✅ Aplicar", font=self.BTN_FONT, bg="#C8E6C9", command=self.apply_change_permanent).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="✂️ Eliminar Fondo (HSV)", font=self.BTN_FONT, bg="#E1BEE7", command=self.remove_bg_hsv).pack(fill=tk.X, pady=2)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)

        # 4. OBJETIVO
        tk.Label(control_frame, text="4. Objetivo: Detección Hojas", font=self.HEADER_FONT, bg="#f0f0f0", fg="#4A148C").pack(anchor="w")
        tk.Label(control_frame, text="Área Mínima (Núcleos):", bg="#f0f0f0").pack(anchor="w")
        tk.Spinbox(control_frame, from_=50, to=10000, increment=50, textvariable=self.min_area_var).pack(fill=tk.X)

        tk.Button(control_frame, text="🎯 Objetivo: Análisis Completo", font=("Segoe UI", 11, "bold"), bg="#673AB7", fg="white", pady=10, 
                  command=self.run_objective_analysis).pack(fill=tk.X, pady=10)
        
        tk.Label(control_frame, text="Detecta hojas verdes y marrones.\nVer reporte en ventana emergente.", bg="#f0f0f0", fg="#666").pack(fill=tk.X)

        # PANEL DERECHO
        display_frame = tk.Frame(self.root, bg="white")
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.lbl_current = tk.Label(display_frame, text="Original", bg="#E0E0E0")
        self.lbl_current.place(relx=0.02, rely=0.05, relwidth=0.47, relheight=0.8)
        self.lbl_preview = tk.Label(display_frame, text="Resultado", bg="#E0E0E0")
        self.lbl_preview.place(relx=0.51, rely=0.05, relwidth=0.47, relheight=0.8)

    # --- FUNCIONES ---
    def load_main_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_img = img
        self.current_img = img
        self.show_image(self.current_img, self.lbl_current)

    def load_secondary_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        messagebox.showinfo("Info", "Imagen secundaria cargada.")

    def show_image(self, img_arr, label_widget):
        if img_arr is None: return
        pil_image = Image.fromarray(img_arr)
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        if w < 10: w, h = 400, 300
        pil_image.thumbnail((w, h))
        tk_image = ImageTk.PhotoImage(pil_image)
        label_widget.config(image=tk_image, text="")
        label_widget.image = tk_image

    def process_algorithm(self, input_img):
        result = input_img.copy()
        k = self.kernel_var.get()
        if k % 2 == 0: k += 1
        scalar = self.scalar_var.get()
        noise_pct = self.noise_var.get() / 100.0
        gamma = self.scale_gamma.get() / 10.0

        filt = self.combo_filters.get()
        if filt != "Ninguno": result = ImageProcessor.apply_filters(result, filt, k)

        op = self.combo_arithmetic.get()
        if "Img2" in op and self.img2 is not None:
            img2_r = cv2.resize(self.img2, (result.shape[1], result.shape[0]))
            if "Sumar" in op: result = cv2.addWeighted(result, 0.5, img2_r, 0.5, 0)
            elif "Restar" in op: result = cv2.subtract(result, img2_r)
            elif "Multiplicar" in op: result = cv2.multiply(result, img2_r.astype(np.float32)/255.0).astype(np.uint8)
            elif "AND" in op: result = cv2.bitwise_and(result, img2_r)
            elif "OR" in op: result = cv2.bitwise_or(result, img2_r)
            elif "XOR" in op: result = cv2.bitwise_xor(result, img2_r)
        elif "Escalar" in op:
            if "Sumar" in op: result = cv2.add(result, (scalar, scalar, scalar, 0))
            elif "Restar" in op: result = cv2.subtract(result, (scalar, scalar, scalar, 0))
        elif "NOT" in op: result = cv2.bitwise_not(result)

        ns = self.combo_noise.get()
        if "Gaussiano" in ns:
             gauss = np.random.normal(0, noise_pct * 255, result.shape).astype('uint8')
             result = cv2.add(result, gauss)
        elif "Sal y Pimienta" in ns:
             probs = np.random.random(result.shape[:2])
             result[probs < (noise_pct/2)] = 0
             result[probs > (1 - noise_pct/2)] = 255

        bri = self.combo_brightness.get()
        if bri != "Ninguno": result = ImageProcessor.apply_brightness(result, bri, gamma)
            
        seg = self.combo_segmentation.get()
        if seg != "Ninguno": result = ImageProcessor.apply_segmentation(result, seg)
            
        return result

    def remove_bg_hsv(self):
        if self.current_img is None: return
        self.preview_img_data = ImageProcessor.remove_background_hsv(self.current_img)
        self.show_image(self.preview_img_data, self.lbl_preview)

    def preview_process(self):
        if self.current_img is None: return
        self.preview_img_data = self.process_algorithm(self.current_img)
        self.show_image(self.preview_img_data, self.lbl_preview)

    def apply_change_permanent(self):
        if self.preview_img_data is not None:
            self.current_img = self.preview_img_data
            self.show_image(self.current_img, self.lbl_current)
            self.preview_img_data = None

    def reset_image(self):
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self.show_image(self.current_img, self.lbl_current)

    def open_report_window(self, stats):
        """Abre una ventana secundaria con el reporte detallado y scrollbar."""
        report_win = tk.Toplevel(self.root)
        report_win.title("Reporte Detallado de Hojas")
        report_win.geometry("500x600")

        lbl_summary = tk.Label(report_win, text=f"RESUMEN:\nTotal: {stats['total']} | Sanas: {stats['healthy']} | Dañadas: {stats['bad']}\nError: {stats['error_pct']}%", 
                               font=("Consolas", 12, "bold"), bg="#e1e1e1", pady=10)
        lbl_summary.pack(fill=tk.X)

        # Widget de texto con scrollbar
        txt_area = scrolledtext.ScrolledText(report_win, width=60, height=30, font=("Consolas", 10))
        txt_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Construir tabla de texto
        header = f"{'ID':<4} {'ESTADO':<10} {'MOTIVO':<12} {'HUE':<6} {'ÁREA'}"
        txt_area.insert(tk.END, header + "\n")
        txt_area.insert(tk.END, "-"*50 + "\n")

        for l in stats['details']:
            row = f"{l['id']:<4} {l['status']:<10} {l['reason']:<12} {l['hue']:<6} {l['area']}\n"
            txt_area.insert(tk.END, row)

        # Resaltar colores (Opcional, simple)
        txt_area.tag_config('danger', foreground='red')
        
        # Bloquear edición
        txt_area.configure(state='disabled')

    def run_objective_analysis(self):
        if self.current_img is None:
            messagebox.showerror("Error", "Carga imagen")
            return
        
        min_area = self.min_area_var.get()
        final, steps, stats, hist = ImageProcessor.advanced_leaf_analysis(self.current_img, min_area)
        
        # 1. Abrir ventana de reporte con Scrollbar
        self.open_report_window(stats)

        # 2. Visualización Matplotlib (Gráficos)
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 4)

        titles = ["1. Original", "2. Máscara (Verde+Marrón)", "3. Erosión (Separación)", "4. Detección Final"]
        for i, img in enumerate(steps):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img)
            ax.set_title(titles[i])
            ax.axis('off')

        ax_hist = fig.add_subplot(gs[1, :])
        ax_hist.plot(hist, color='g')
        ax_hist.fill_between(range(len(hist)), hist.flatten(), color='g', alpha=0.3)
        ax_hist.set_title("Histograma de Color (Hue) - Hojas Detectadas")
        ax_hist.set_xlabel("Valor Hue (0-180)")
        
        plt.tight_layout()
        plt.show()