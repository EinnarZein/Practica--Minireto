#Importar librerias necesarias 📚
import cv2
import numpy as np
from scipy.ndimage import generic_filter

class ImageProcessor:
    """
    Contiene TODA la lógica matemática de procesamiento de imágenes.
    Recupera todas las funciones del sistema original.
    """
    # --- FUNCIONES GENERALES DE PROCESAMIENTO DE IMÁGENES ---
    @staticmethod
    def apply_filters(img, filter_name, k_size):
        # Asegurar copia y versión en gris
        res = img.copy()
        gray_result = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) if len(res.shape) == 3 else res

        # --- FILTROS BÁSICOS ---
        #--ESCALA DE GRISES---
        if filter_name == "Escala de Grises" and len(res.shape) == 3:
            return cv2.cvtColor(cv2.cvtColor(res, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        #--FILTRO PROMEDIADOR---
        elif filter_name == "Filtro Promediador":
            return cv2.blur(res, (k_size, k_size))
        #--FILTRO GAUSSIANO---
        elif filter_name == "Filtro Gaussiano":
            return cv2.GaussianBlur(res, (k_size, k_size), 0)
        #--FILTRO MEDIANA---
        elif filter_name == "Filtro Mediana":
            return cv2.medianBlur(res, k_size)
        
        # --- FILTROS MORFOLÓGICOS / ESTADÍSTICOS ---
        #--FILTRO MÍNIMO Y MÁXIMO---
        elif filter_name == "Filtro Mínimo":
            return cv2.erode(res, np.ones((k_size, k_size), np.uint8))
        #--FILTRO MÁXIMO---
        elif filter_name == "Filtro Máximo":
            return cv2.dilate(res, np.ones((k_size, k_size), np.uint8))
        #--FILTRO MODA---
        elif filter_name == "Filtro Moda":
            try:
                fast_mode = lambda x: np.bincount(x.astype(int)).argmax()
                # Aplicamos sobre gris para velocidad, luego convertimos
                gray_res = generic_filter(gray_result, fast_mode, size=k_size, mode='mirror')
                return cv2.cvtColor(gray_res.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            except: return res
        #--FILTRO PROMEDIO PONDERADO (GAUSSIANO 1D)---
        elif filter_name == "Filtro Prom. Pesado":
            k1d = cv2.getGaussianKernel(k_size, -1)
            return cv2.filter2D(res, -1, k1d * k1d.T)

        # --- FILTROS DE BORDES ---
        #--FILTRO CANNY---
        elif filter_name == "Bordes Canny":
            return cv2.cvtColor(cv2.Canny(gray_result, 100, 200), cv2.COLOR_GRAY2RGB)
        #--FILTRO SOBEL---
        elif filter_name == "Filtro Sobel":
            sobel = cv2.magnitude(cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=k_size), 
                                  cv2.Sobel(gray_result, cv2.CV_64F, 0, 1, ksize=k_size))
            return cv2.cvtColor(cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        #--FILTRO PREWITT---
        elif filter_name == "Filtro Prewitt":
            kx, ky = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]), np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            prewitt = cv2.magnitude(cv2.filter2D(gray_result, cv2.CV_64F, kx), cv2.filter2D(gray_result, cv2.CV_64F, ky))
            return cv2.cvtColor(cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        #--FILTRO ROBERTS CROSS---
        elif filter_name == "Filtro Robert":
            roberts_cross_v = np.array( [[1, 0 ], [0,-1 ]] )
            roberts_cross_h = np.array( [[ 0, 1 ], [ -1, 0 ]] )
            img_robert = cv2.magnitude(cv2.filter2D(gray_result, cv2.CV_64F, roberts_cross_v), 
                                       cv2.filter2D(gray_result, cv2.CV_64F, roberts_cross_h))
            return cv2.cvtColor(cv2.normalize(img_robert, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        #--FILTRO KIRSCH (SIMPLIFICADO A UNA DIRECCIÓN)---
        elif filter_name == "Filtro Kirsch":
             # Implementación básica Kirsch (Norte)
             k_kirsch = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
             kirsch = cv2.filter2D(gray_result, cv2.CV_64F, k_kirsch)
             return cv2.cvtColor(cv2.normalize(kirsch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        #--FILTRO LAPLACIANO (DOS VERSIONES)---
        elif filter_name == "Filtro Laplaciano (Opencv)":
            lap = cv2.Laplacian(gray_result, cv2.CV_64F, ksize=k_size)
            return cv2.cvtColor(cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Laplaciano (8 Vecinos)":
            k_lap = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            lap = cv2.filter2D(gray_result, cv2.CV_64F, k_lap)
            return cv2.cvtColor(cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        #--NEGATIVO---
        elif filter_name == "Negativo":
            return cv2.bitwise_not(res)
        
        return res

    # --- FUNCIONES DE BRILLO Y CONTRASTE ---
    @staticmethod
    def apply_brightness(img, method, gamma_param):
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            gray = cv2.split(ycrcb)[0]
        else:
            gray, ycrcb = img, None
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf_normalized = hist.cumsum() / hist.cumsum().max()
        new_gray = gray.copy()

        if method == "Ecualización Uniforme":
            new_gray = cv2.equalizeHist(gray)
        elif method == "Ecualización Exponencial":
            cdf_normalized = np.minimum(cdf_normalized, 0.9999) 
            transformed = -1 * np.log(1 - cdf_normalized) / 0.05
            new_gray = ((transformed / transformed.max()) * 255).astype(np.uint8)[gray]
        elif method == "Ecualización Rayleigh":
            cdf_normalized = np.minimum(cdf_normalized, 0.9999) 
            transformed = 0.5 * np.sqrt(-2 * np.log(1 - cdf_normalized))
            new_gray = ((transformed / transformed.max()) * 255).astype(np.uint8)[gray]
        elif method == "Ecualización Hiperbólica":
             # Implementación básica simulada
             new_gray = cv2.equalizeHist(gray) 
        elif method == "Ecualización Hipercúbica":
             new_gray = cv2.equalizeHist(gray)
        elif method == "Función Potencia (Gamma)":
            new_gray = np.uint8(np.power(gray / 255.0, gamma_param) * 255)
        elif method == "Corrección Gamma (Auto)":
            mean = np.mean(gray) / 255.0
            gamma = np.log(0.5) / np.log(mean + 1e-6)
            new_gray = np.uint8(np.power(gray / 255.0, gamma) * 255)

        if ycrcb is not None:
            channels = list(cv2.split(ycrcb))
            channels[0] = new_gray
            return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2RGB)
        return cv2.cvtColor(new_gray, cv2.COLOR_GRAY2RGB)

    # --- FUNCIONES DE SEGMENTACIÓN ---
    @staticmethod
    def apply_segmentation(img, method):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        #---UMBRALADO OTSU---
        if method == "Umbralado Otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        #--UMBRALADO KAPUR (ENTROPÍA)---
        elif method == "Umbralado Kapur (Entropía)":
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum()
            cdf = hist.cumsum()
            max_ent, best_thresh = 0, 0
            for t in range(1, 255):
                w0, w1 = cdf[t], 1 - cdf[t]
                if w0 == 0 or w1 == 0: continue
                p0, p1 = hist[:t+1]/w0, hist[t+1:]/w1
                h0, h1 = -np.sum(p0 * np.log(p0 + 1e-10)), -np.sum(p1 * np.log(p1 + 1e-10))
                if h0 + h1 > max_ent: max_ent, best_thresh = h0 + h1, t
            _, thresh = cv2.threshold(gray, best_thresh, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        #--UMBRALADO VALLE DEL HISTOGRAMA---
        elif method == "Mínimo del Histograma":
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            kernel = cv2.getGaussianKernel(9, 2)
            hist_smooth = cv2.filter2D(hist, -1, kernel).flatten()
            peaks = [i for i in range(1, 255) if hist_smooth[i-1] < hist_smooth[i] and hist_smooth[i+1] < hist_smooth[i]]
            val_thresh = 127
            if len(peaks) >= 2:
                p1, p2 = peaks[0], peaks[-1]
                valley_region = hist_smooth[p1:p2]
                if len(valley_region) > 0: val_thresh = p1 + np.argmin(valley_region)
            _, thresh = cv2.threshold(gray, val_thresh, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        #--MULTIUMBRALIZACIÓN K-MEANS---
        elif method == "Multiumbralización (K-Means)":
            Z = np.float32(gray.reshape((-1, 1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, label, center = cv2.kmeans(Z, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            res = np.uint8(center)[label.flatten()].reshape((gray.shape))
            return cv2.applyColorMap(res, cv2.COLORMAP_JET)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # --- FUNCIONES ENFOCADAS AL PROYECTO FINAL ---
    '''
    Se elimina el fondo de la imagen usando el espacio de color HSV para segmentar colores específicos.
    En este caso permite dejar visible solo el objeto verde sobre un fondo irregular.
    ¿Como funciona?
    1. Convierte la imagen a espacio HSV.
    2. Crea una máscara para los tonos verdes.
    3. Aplica operaciones morfológicas para limpiar la máscara.
    4. Devuelve la imagen con el fondo eliminado (área no verde).
    '''
    @staticmethod
    def remove_background_hsv(img):
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.bitwise_and(img, img, mask=mask)

    '''
    Detecta objetos en la imagen basándose en contornos y dibuja sus contornos junto con anotaciones de área y perímetro.
    Retorna la imagen anotada y el conteo de objetos detectados.
    '''
    # --- DETECCIÓN DE OBJETOS Y ANOTACIONES ---
    @staticmethod
    def detect_objects(img, min_area):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        count = 0
        h_img, w_img = output.shape[:2]
        font_scale = max(0.6, w_img / 1200.0) 
        thickness = max(2, int(font_scale * 3))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                count += 1
                perimeter = cv2.arcLength(cnt, True)
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), thickness)
                x,y,w,h = cv2.boundingRect(cnt)
                info = f"A:{int(area)} P:{int(perimeter)}"
                (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(output, (x, y - th - 10), (x + tw, y), (0,0,0), -1)
                cv2.putText(output, info, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
        return output, count