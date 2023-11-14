import random
import numpy as np


class Red:
    """Se manda una lista de números que describa los numeros de neuronas
    que contiene cada capa según el índice"""

    def __init__(self, estructura:list, tasaA=3.0, activacion='sigmoide'):
        self.estructura = estructura
        """La función zip() concatena de acuerdo a los indices en tuplas, sacando 'x' y 'y'
        Se toman las tuplas a manera de generar matrices que representen los pesos entre las capas de neuronas"""
        self.pesos = [np.random.randn(x, y) for x, y in zip(estructura[1:], estructura[:-1])]
        # Genera una lista de arreglos a partir de la segunda capa, siendo el BIAS o Sesgo de cada neurona:
        self.sesgos = [np.random.randn(x, 1) for x in estructura[1:]]
        if activacion == 'sigmoide':
            self.activacion = sigmoide
            self.aPrima = derivada_sigmoide
        elif activacion == 'tanh':
            self.activacion = tanh
            self.aPrima = derivada_tanh
        self.tasaAprendizaje = tasaA
        self.epochs = None
        self.numCapas = len(estructura)
        self.errores = list()

    def clasifica(self, entrada):
        capa = entrada
        for p,s in zip(self.pesos,self.sesgos):
            z = np.dot(p,capa)+s
            capa = self.activacion(z)
        return capa

    def retropropagacion(self,entrada,etiqueta):
        activacionCapa = entrada
        activaciones = list()
        activaciones.append(activacionCapa)
        zetas = list()

        for p,s in zip(self.pesos,self.sesgos):
            z = np.dot(p,activacionCapa)+s#CHECA
            zetas.append(z)
            activacionCapa = self.activacion(z)
            activaciones.append(activacionCapa)

        error = (activaciones[-1]-etiqueta)
        self.errores.append(np.mean(error))

        delta = error * self.aPrima(zetas[-1])
        nablaP = [np.zeros(p.shape) for p in self.pesos]
        nablaS = [np.zeros(s.shape) for s in self.sesgos]

        nablaS[-1]=delta
        nablaP[-1]=np.dot(delta,activaciones[-2].T)

        for capa in range(2,self.numCapas):
            delta = np.dot(self.pesos[1-capa].T , delta) * self.aPrima(zetas[-capa])
            nablaS[-capa] = delta
            nablaP[-capa] = np.dot(delta,activaciones[-1-capa].T)

        return (nablaP,nablaS)

    def actualizaPyS(self,miniLote):

        sumNP = [np.zeros(p.shape) for p in self.pesos]
        sumNS = [np.zeros(s.shape) for s in self.sesgos]
        for entrada, etiqueta in miniLote:
            bpNablaP,bpNablaS = self.retropropagacion(entrada,etiqueta)
            sumNP = [sP+nP for sP, nP in zip(sumNP,bpNablaP)] #Sumatoria
            sumNS = [sS+nS for sS, nS in zip(sumNS,bpNablaS)]

        prom = self.tasaAprendizaje/len(miniLote)
        self.pesos = [p-(prom)*snP for p,snP in zip(self.pesos,sumNP)]
        self.sesgos = [s-(prom)*snS for s,snS in zip(self.sesgos,sumNS)]

    def entrenaDescensoEstocastico(self,dataEnt, tamMiniLote,epochs=1):
        self.epochs = epochs
        for x in range(self.epochs):
            random.shuffle(dataEnt)
            miniLotes = [dataEnt[i:i+tamMiniLote] for i in range(0,len(dataEnt),tamMiniLote)]
            for m in miniLotes:
                self.actualizaPyS(m)
            print("Epoca: ",x)

    def entrena(self,entrada,etiqueta,tA=None):
        if tA != None:
            self.tasaAprendizaje=tA
        nablaPesos,nablaSesgo = self.retropropagacion(entrada,etiqueta)
        self.pesos = [p-self.tasaAprendizaje*nP for p,nP in zip(self.pesos,nablaPesos)]
        self.sesgos = [s-self.tasaAprendizaje*nS for s,nS in zip(self.sesgos, nablaSesgo)]

def sigmoide(x):
    return 1.0/(1.0 + np.exp(-x))
def derivada_sigmoide(x):
    return sigmoide(x)*(1-sigmoide(x))
def tanh(x):
    return np.tanh(x)
def derivada_tanh(x):
    return (1.0 - (tanh(x)**2))

"""
Algoritmos de optimización:
Para conjuntos de datos grandes: adam o rmsprop (algoritmos optimización)

MAIN:"""
