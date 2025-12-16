
import cv2
import numpy as np
from scipy.ndimage import generic_filter

class ImageProcessor:
    """
    Lógica de procesamiento de imágenes para la aplicación Mini Reto PDI.
    Proporciona métodos para aplicar filtros, ajustes de brillo, segmentación
    y análisis avanzado de hojas en imágenes.
    """

    # --- 1. FILTROS GENERALES (Sin cambios) ---
    @staticmethod
    def apply_filters(img, filter_name, k_size):
        res = img.copy()
        gray_result = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) if len(res.shape) == 3 else res

        if filter_name == "Escala de Grises" and len(res.shape) == 3:
            return cv2.cvtColor(cv2.cvtColor(res, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Promediador":
            return cv2.blur(res, (k_size, k_size))
        elif filter_name == "Filtro Gaussiano":
            return cv2.GaussianBlur(res, (k_size, k_size), 0)
        elif filter_name == "Filtro Mediana":
            return cv2.medianBlur(res, k_size)
        elif filter_name == "Filtro Mínimo":
            return cv2.erode(res, np.ones((k_size, k_size), np.uint8))
        elif filter_name == "Filtro Máximo":
            return cv2.dilate(res, np.ones((k_size, k_size), np.uint8))
        elif filter_name == "Filtro Moda":
            try:
                fast_mode = lambda x: np.bincount(x.astype(int)).argmax()
                gray_res = generic_filter(gray_result, fast_mode, size=k_size, mode='mirror')
                return cv2.cvtColor(gray_res.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            except: return res
        elif filter_name == "Filtro Prom. Pesado":
             k1d = cv2.getGaussianKernel(k_size, -1)
             return cv2.filter2D(res, -1, k1d * k1d.T)
        elif filter_name == "Bordes Canny":
            return cv2.cvtColor(cv2.Canny(gray_result, 100, 200), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Sobel":
            s_x = cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=k_size)
            s_y = cv2.Sobel(gray_result, cv2.CV_64F, 0, 1, ksize=k_size)
            sobel = cv2.magnitude(s_x, s_y)
            return cv2.cvtColor(cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Prewitt":
            kx, ky = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]), np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            prewitt = cv2.magnitude(cv2.filter2D(gray_result, cv2.CV_64F, kx), cv2.filter2D(gray_result, cv2.CV_64F, ky))
            return cv2.cvtColor(cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Robert":
            r_v, r_h = np.array([[1, 0],[0,-1]]), np.array([[0, 1],[-1, 0]])
            robert = cv2.magnitude(cv2.filter2D(gray_result, cv2.CV_64F, r_v), cv2.filter2D(gray_result, cv2.CV_64F, r_h))
            return cv2.cvtColor(cv2.normalize(robert, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Kirsch":
             k_kirsch = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
             kirsch = cv2.filter2D(gray_result, cv2.CV_64F, k_kirsch)
             return cv2.cvtColor(cv2.normalize(kirsch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Laplaciano (Opencv)":
            lap = cv2.Laplacian(gray_result, cv2.CV_64F, ksize=k_size)
            return cv2.cvtColor(cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Filtro Laplaciano (8 Vecinos)":
            k_lap = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
            lap = cv2.filter2D(gray_result, cv2.CV_64F, k_lap)
            return cv2.cvtColor(cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif filter_name == "Negativo":
            return cv2.bitwise_not(res)
        
        return res

    # --- 2. BRILLO Y SEGMENTACIÓN ---
    @staticmethod
    def apply_brightness(img, method, gamma_param):
        if len(img.shape) == 3:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            gray = cv2.split(ycrcb)[0]
        else: gray, ycrcb = img, None
        
        new_gray = gray.copy()
        if method == "Ecualización Uniforme":
            new_gray = cv2.equalizeHist(gray)
        elif method == "Ecualización Exponencial":
             hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
             cdf_normalized = hist.cumsum() / hist.cumsum().max()
             cdf_normalized = np.minimum(cdf_normalized, 0.9999) 
             transformed = -1 * np.log(1 - cdf_normalized) / 0.05
             new_gray = ((transformed / transformed.max()) * 255).astype(np.uint8)[gray]
        elif method == "Ecualización Rayleigh":
             hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
             cdf_normalized = hist.cumsum() / hist.cumsum().max()
             cdf_normalized = np.minimum(cdf_normalized, 0.9999) 
             transformed = 0.5 * np.sqrt(-2 * np.log(1 - cdf_normalized))
             new_gray = ((transformed / transformed.max()) * 255).astype(np.uint8)[gray]
        elif method == "Ecualización Hiperbólica":
              new_gray = np.uint8(255 * np.log1p(gray) / np.log1p(255))
        elif method == "Ecualización Hipercúbica":
              new_gray = np.uint8(255 * np.cbrt(gray / 255.0))
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

    @staticmethod
    def apply_segmentation(img, method):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        if method == "Umbralado Otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        elif method == "Umbralado Kapur (Entropía)":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        elif method == "Mínimo del Histograma":
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        elif method == "Multiumbralización (K-Means)":
            Z = np.float32(gray.reshape((-1, 1)))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, label, center = cv2.kmeans(Z, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            res = np.uint8(center)[label.flatten()].reshape((gray.shape))
            return cv2.applyColorMap(res, cv2.COLORMAP_JET)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def remove_background_hsv(img):
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
        lower_gen = np.array([10, 30, 30])
        upper_gen = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_gen, upper_gen)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.bitwise_and(img, img, mask=mask)

    # --- 3. PROYECTO OBJETIVO (AJUSTADO PARA DETECTAR MÁS HOJAS Y MENOS SUELO) ---
    @staticmethod
    def advanced_leaf_analysis(img, min_area=300):
        # 1. Preprocesamiento
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

        # 2. MÁSCARAS AJUSTADAS (EL TRUCO ESTÁ EN EL 'VALUE' O BRILLO)
        
        # Máscara A: Hojas Verdes (Sanas)
        # H: 30-90, S: 40-255, V: 40-255
        lower_green = np.array([30, 40, 40]) 
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Máscara B: Hojas Marrones/Amarillas (Enfermas)
        # CRÍTICO: El suelo es Hue 10-20 pero Value < 60. Las hojas son Value > 60.
        # Subimos el mínimo de 'Value' a 60 para ignorar el suelo oscuro.
        # Subimos el mínimo de 'Saturation' a 50 para ignorar el suelo grisáceo.
        lower_brown = np.array([5, 50, 60])
        upper_brown = np.array([30, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # Combinamos las dos máscaras
        mask_combined = cv2.bitwise_or(mask_green, mask_brown)

        # 3. Limpieza Morfológica
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        opening = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        
        # 4. EROSIÓN SUAVE (MENOS AGRESIVA PARA NO BORRAR HOJAS PEQUEÑAS)
        kernel_erosion = np.ones((3,3), np.uint8) # Kernel más pequeño 
        eroded_mask = cv2.erode(opening, kernel_erosion, iterations=3) 

        # 5. Análisis de Conectividad (Vecindad 8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)

        output_img = img.copy()
        leaf_data = []
        count_healthy = 0
        count_bad = 0
        final_mask_for_hist = np.zeros_like(mask_combined)

        # Empezamos desde 1 para ignorar el fondo (0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Si el núcleo erosionado es muy pequeño, lo ignoramos
            if area < (min_area / 3): continue 

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Reconstruimos la hoja un poco (Dilatación local para recuperar tamaño real)
            # Esto es visual, usamos el núcleo para contar pero queremos ver la hoja entera
            leaf_mask_local = np.zeros_like(mask_combined)
            leaf_mask_local[labels == i] = 255
            # Dilatamos la máscara local para recuperar el área perdida por la erosión
            leaf_mask_local = cv2.dilate(leaf_mask_local, kernel_erosion, iterations=3)

            # Recalculamos área real aproximada
            contours, _ = cv2.findContours(leaf_mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            real_area = cv2.contourArea(contours[0]) if contours else area

            # Análisis de color en la zona
            mean_color = cv2.mean(hsv, mask=leaf_mask_local)
            hue_val = mean_color[0]
            sat_val = mean_color[1]
            val_val = mean_color[2]

            # FILTRO DE SEGURIDAD CONTRA TIERRA
            # Si es marrón (hue < 30) pero muy oscuro (val < 55) o muy gris (sat < 40), es tierra -> IGNORAR
            if hue_val < 35 and (val_val < 55 or sat_val < 40):
                continue

            # Clasificación
            status = "Sana"
            color_box = (0, 255, 0) # Verde
            reason = "Ok"

            if hue_val < 35 or hue_val > 100:
                status = "Dañada"
                reason = "Color Irreg."
                color_box = (255, 0, 0)
                count_bad += 1
            else:
                count_healthy += 1
                final_mask_for_hist = cv2.bitwise_or(final_mask_for_hist, leaf_mask_local)

            # Dibujamos
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color_box, 2)
            cv2.putText(output_img, f"{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

            leaf_data.append({
                "id": i,
                "status": status,
                "reason": reason,
                "hue": int(hue_val),
                "area": int(real_area)
            })

        # Estadísticas
        total = count_healthy + count_bad
        error_pct = (count_bad / total * 100) if total > 0 else 0
        
        global_stats = {
            "total": total,
            "healthy": count_healthy,
            "bad": count_bad,
            "error_pct": round(error_pct, 2),
            "details": leaf_data
        }

        hist_hue = cv2.calcHist([hsv], [0], final_mask_for_hist, [180], [0, 180])
        steps = [img, cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB), cv2.cvtColor(eroded_mask, cv2.COLOR_GRAY2RGB), output_img]
        
        return output_img, steps, global_stats, hist_hue