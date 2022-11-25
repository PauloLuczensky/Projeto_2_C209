#Subtração de Fundo

#Podemos usar a cvzone biblioteca para remover o fundo de uma imagem 
#que usa a mediapipe biblioteca para remover o fundo
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


segmentor = SelfiSegmentation()

#Abrindo a imagem por meio da biblioteca cv2
#img = cv2.imread('ex_blend1.jpg')
img = np.array(Image.open('ex_blend1.jpg'))

# OpenCV abre a imagem em BRG,
# mas ela deve estar em RGB 
# para transformá-la em escala de cinza
# Utiliza-se funções próprias da cv2 para realizar a alteração
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Em vez de realizar a tranformaçãos pela função do cva
#realizou-se por meio do grayscale ensinado no laboratório
def grayscale(img_np):
    (l, c, p) = img_np.shape

    img_avg = np.zeros(shape=(l, c), dtype=np.uint8)
    for i in range(l):
        for j in range(c):
            r = float(img_np[i, j, 0])
            g = float(img_np[i, j, 1])
            b = float(img_np[i, j, 2])
        
            img_avg[i, j] = (r + g + b) / 3
     
    return img_avg

img_avg = grayscale(img)

#Porém, converteu-se em RGB usando uma função do cv2
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Listando os argumentos da função removeBG
# 1º) imagem de entrada; 
# 2º) cor que queremos usar como a nova cor de fundo.
# 3º) limite que podemos definir de acordo com nossa imagem fornecida 
img_Out = segmentor.removeBG(img_rgb, (255,255,255), threshold=0.99)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(img_Out)

cv2.imshow('img',img_Out)

#A função waitkey() do Python OpenCV permite que os usuários exibam uma janela por determinados milissegundos 
#ou até que qualquer tecla seja pressionada.
cv2.waitKey(0)

#A função destroyAllWindows()  permite que os usuários destruam ou 
#fechem todas as janelas a qualquer momento após sair do script
cv2.destroyAllWindows()