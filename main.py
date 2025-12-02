'''
Aqui va el codigo principal de la aplicacion 👑
'''

import tkinter as tk
from src.gui import ImageEditorApp

def main():
    root = tk.Tk()
    
    # Configuracion de la ventana
    root.iconbitmap('assets/img/icono.ico')
    app = ImageEditorApp(root)
    root.mainloop()

#llamada al main
if __name__ == "__main__":
    main()