'''
Inicialización del paquete src 📦
'''

from .logic import ImageProcessor
from .gui import ImageEditorApp

# Definimos qué se exporta si alguien usa "from src import *"
__all__ = ['ImageProcessor', 'ImageEditorApp']