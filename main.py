import funcionesIMG as img
import AnalizadorRNA as red
import string
import random
import matplotlib.pyplot as plt
import matriz as mat

trainingData = img.dataLoaderBrailleDataset(20,"BrailleDataset",26)
random.shuffle(trainingData)
#data = img.dataLoaderTrainData(20,"BrailleDataset",26)
trainData = trainingData[0::2]
mitad = trainingData[1::2]

plus = mitad[1::2]
test = mitad[0::2]
trainData.extend(plus)

letras = string.ascii_lowercase
puntos = range(6)

estructura = [784,100,100,6]
tasaAprendizaje = 0.1
epochs = 100
tamañoLote = 15


RNA = red.Red(estructura,tasaAprendizaje)
RNA.entrenaDescensoEstocastico(trainData,tamañoLote,epochs)

def imprimeClass(test,numejemplos:int):
    random.shuffle(test)
    for i in range(numejemplos):
        x, y = test[i]
        res = RNA.clasifica(x)
        print("\nResultado Esperado:\n", y, "\n")
        print("Resultado Obtenido:")
        for i, l in zip(res, puntos):
            print(l, "%.3f" % i)

imprimeClass(test,4)

testData = img.dataLoaderTestData("TestData",26)
print("Nunca antes visto...........")
imprimeClass(testData,4)

def imprimePromError(test_data:list):
    errores = list()
    suma = 0
    random.shuffle(test_data)
    for img,valor in test_data:
        res = RNA.clasifica(img)
        error = valor-res
        suma += error
        errores.append(error)
    promedio = suma/len(errores)
    return promedio,errores

#testData = img.dataLoaderTestData("TestData",26)
#print("Nunca antes visto...........")
#imprimeClass(testData,4)

"""
x,y = mitad[0]

res = RNA.clasifica(x)
print("Resultado Esperado:\n",y,"\n")
print("Resultado Obtenido:")
for i, l in zip(res, puntos):
    print(l, "%.3f" % i)

matriz_evo = mat.matriz(26)
random.shuffle(data)
trainData = data[0::2]
testData = data[1::2]
random.shuffle(testData)
for d,v in trainData:
    matriz_evo.agrega_ejemplo(d,v)
print(matriz_evo.clasifica(testData[0][0]),testData[0][1])
"""