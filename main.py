import cv2
import numpy as np

# Carregar imagem
img = cv2.imread('john1.jpg')

# Verificar se a imagem foi carregada
if img is None:
    print("Erro: Imagem não encontrada!")
else:
    # Mostrar imagem
    cv2.imshow('Minha Imagem', img)
    cv2.waitKey(0)  # Espera até uma tecla ser pressionada
    cv2.destroyAllWindows()