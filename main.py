# =============================================================================
# UBICACIÓN: main.py (Raíz del proyecto)
# =============================================================================
'''
Aqui va el codigo principal de la aplicacion 👑
'''

import tkinter as tk
from tkinter import messagebox
import os

# Importamos la clase de la GUI desde el paquete src
try:
    from src.gui import ImageEditorApp
except ImportError as e:
    # Captura error si la carpeta src o gui.py no existen
    print(f"Error crítico: {e}")
    exit()

def main():
    root = tk.Tk()
    root.title("Mini Reto PDI")
    
    # Configuracion de la ventana
    # Bloque try/except por si no tienes el archivo .ico creado aún
    try:
        if os.path.exists('assets/img/icon.ico'):
            root.iconbitmap('assets/img/icon.ico')
    except Exception:
        pass # Si falla el icono, la app sigue funcionando
    
    # Instanciamos la aplicación
    app = ImageEditorApp(root)
    
    root.mainloop()

# llamada al main
if __name__ == "__main__":
    main()