import PIL.Image
from PIL import Image
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt

def convierteImagen(ruta:str): # convierte la imagen a una Escala de Grises (vector)
    file = Image.open(ruta)
    rgb_img = file.convert('RGB')
    ancho, alto = rgb_img.size
    grises = list()
    for y in range(0,alto):
        for x in range(0,ancho):
            r,g,b = rgb_img.getpixel((x,y))
            gris = float(float(float(r+g+b)/3.0)/255)
            grises.append(gris)
    tam = len(grises)
    patron = np.array(np.zeros((tam,1),float))
    for i in range(0,tam):
        patron[i,0] = float(grises[i])
    return patron

def traspuesta(vector:list,ancho:int,altura:int):
    temp=list()
    for i in range(ancho):
        for j in range(altura):
            temp.append(vector[j*ancho + i ])
    return temp

def cambiaTam(path:str,sizex:int,sizey:int):
    archivo = Image.open(path)
    archivo = archivo.resize((sizex,sizey),PIL.Image.BICUBIC)
    archivo.save("resized_"+path)

def dataLoaderBrailleDataset(numejemplos:int, ruta:str, numLetras:int):
    letras=string.ascii_lowercase
    estados = ["dim","rot","whs"]
    data = list()
    etiqueta = etiquetadoBraille()
    contador = 0
    for i in letras:
        for j in range(numejemplos):
            for k in estados:
                imagenRuta = ruta+"/"+i+'1.JPG'+str(j)+k+'.jpg'

                if(k=='dim'):
                    imagen = convierteImagen(imagenRuta)
                    valor = etiqueta[contador]
                    data.append((imagen,valor))
        contador+=1
    return data

def dataLoaderTestData(ruta:str,numLetras=None):
    data = list()
    letras = string.ascii_lowercase
    etiqueta = etiquetadoBraille()
    contador = 0
    for i in letras:
        rutaI = ruta+'/'+i+'.jpg'
        imagen = convierteImagen(rutaI)
        data.append((imagen,etiqueta[contador]))
        contador+=1
    return data

def dataLoaderTrainData(numejemplos:int,ruta:str,numLetras):
    letras = string.ascii_lowercase
    estados = ["dim", "rot", "whs"]
    data = list()
    etiqueta = etiquetado(numLetras)
    contador = 0
    for i in letras:
        for j in range(numejemplos):
            for k in estados:
                imagenRuta = ruta + "/" + i + '1.JPG' + str(j) + k + '.jpg'
                if (k == 'dim'):
                    imagen = convierteImagen(imagenRuta)
                    #valor = etiqueta[contador]
                    data.append((imagen, contador))
        contador +=1
    return data

def etiquetado(n):
    etiquetasL = list()
    for x in range(26):
        etiqueta = np.zeros((n, 1))
        etiqueta[x] = 1.0
        etiquetasL.append(etiqueta)
    return etiquetasL

def segmentar(rutaIMG:str,k:int,intentos:int):
    imagen = cv2.imread(rutaIMG)
    """Se tiene que transformar la imagen a HSV para que el matiz, la luminosidad y la saturación sean relevantes"""
    img = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

    matriz3x3RGB = img.reshape((-1,3))
    matriz3x3RGB = np.float32(matriz3x3RGB)
    criteriosOpenCV = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    """
    cv2.kmeans(muestras,nclusters(k),criterio,intentos,banderas)
    muestras: debe ser tipo np.float32
    k: clusters que se requieren al terminar
    criterio: tipo, máxima iteración y epsilon(specified accuracy)
    """
    ret,etiqueta,centro=cv2.kmeans(matriz3x3RGB,k,None,criteriosOpenCV,intentos,cv2.KMEANS_PP_CENTERS)

    centros =np.uint8(centro)
    print(centros)
    resultado = centros[etiqueta.flatten()]
    imagen_resultante=resultado.reshape((img.shape))
    cv2.imwrite("imgSEG.jpg",imagen_resultante)
"""
    figure_size = 10
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(imagen_resultante)
    plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
    plt.show()
"""

def bordes(rutaIMG:str,limite1=150,limite2=200):
    imagen = cv2.imread(rutaIMG)
    """Se tiene que transformar la imagen a HSV para que el matiz, la luminosidad y la saturación sean relevantes"""
    img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    figure_size = 10
    edges = cv2.Canny(img, limite1, limite2)
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def sumatoria(vector,lim:int):
    lista = list()
    count = 0
    suma=0
    for x in vector:
        suma=suma+x
        count+=1
        if(count==lim):
            lista.append(suma)
            suma=0
            count=0
    return lista

def analisisGrises(rutaIMG="recursos/imgSEG.jpg"):
    file=Image.open(rutaIMG)
    alt= file.height
    anch= file.width
    vector_grises = convierteImagen(rutaIMG)
    traspuesto = traspuesta(vector_grises,anch,alt)

    if(len(vector_grises)!=len(traspuesto)):
        print("error")

    #analisis de altura:
    altura=sumatoria(traspuesto,alt)

    #analisis de anchura:
    longitud=sumatoria(vector_grises,anch)

    return longitud,altura

def plot(vector:list):
    plt.plot(vector)
    plt.show()

"""----MAIN:----
et = etiquetadoBraille()
print(len(et))
segmentar("segmentar.jpg",2,10)
#bordes("segmentar.jpg")
"""
#x,y = analisisGrises()
#plot(x)

"""----MAIN:----"""

def etiquetadoBraille():
    etiquetas = list()

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[2] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[3] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[3] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1)) #G
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))#K
    etiqueta[0] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[2] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))#N
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1)) #R
    etiqueta[0] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[4] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[2] = 1.0
    etiqueta[4] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))#W
    etiqueta[1] = 1.0
    etiqueta[2] = 1.0
    etiqueta[3] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[4] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))
    etiqueta[0] = 1.0
    etiqueta[1] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    etiqueta = np.zeros((6, 1))#Z
    etiqueta[0] = 1.0
    etiqueta[3] = 1.0
    etiqueta[4] = 1.0
    etiqueta[5] = 1.0
    etiquetas.append(etiqueta)

    return etiquetas
