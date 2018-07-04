import numpy
import random
"""
    Classe servant à crée des Neural Networks
    Elle sera au début hardcoder en grande partie
    mais deviendra de plus en plus modulables
"""

class NeuralNetwork:

    #Constructeur
    def __init__(self):
        nb_input = 784
        nb_hidden_nodes = 160
        nb_output = 10

        #Création des matrix de weights et Bias
        self.weights_ih = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_hidden_nodes,nb_input)))
        self.weights_ho = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_output,nb_hidden_nodes)))


        self.bias_ih = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_hidden_nodes,1)))
        self.bias_ho = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_output,1)))

        self.learning_rate = 0.01

    #Fait un guess
    def guess(self,features):
        features = numpy.asmatrix(features)

        #Calcul de Input à Hidden
        calcul_ih = numpy.dot(self.weights_ih,features)
        calcul_ih = numpy.add(calcul_ih,self.bias_ih)
        calcul_ih = self.sigmoid(calcul_ih)

        #Calcul de Hidden à Output
        calcul_ho = numpy.dot(self.weights_ho,calcul_ih)
        calcul_ho = numpy.add(calcul_ho,self.bias_ho)
        calcul_ho = self.sigmoid(calcul_ho)

        return calcul_ho


    def train(self,features,label):
        features = numpy.asmatrix(features)
        labels = numpy.asmatrix(label)

        #Calcul de Input à Hidden
        calcul_ih = numpy.dot(self.weights_ih,features)
        calcul_ih = numpy.add(calcul_ih,self.bias_ih)
        calcul_ih = self.sigmoid(calcul_ih)

        #Calcul de Hidden à Output
        calcul_ho = numpy.dot(self.weights_ho,calcul_ih)
        calcul_ho = numpy.add(calcul_ho,self.bias_ho)
        calcul_ho = self.sigmoid(calcul_ho)

        erreurs = numpy.subtract(labels,calcul_ho)

        #Calcul du gradient
        gradient = numpy.asmatrix(self.dsigmoid(numpy.asarray(calcul_ho)))
        gradient = numpy.multiply(gradient,erreurs)
        gradient = numpy.multiply(gradient,self.learning_rate)

        #Update des bias
        self.bias_ho = numpy.add(self.bias_ho,gradient)

        #Calcul des delta weights
        transposed_ih = numpy.transpose(calcul_ih)
        delta_weights_ho = numpy.multiply(gradient,transposed_ih)

        self.weights_ho = numpy.add(self.weights_ho,delta_weights_ho)

        #Backpropagation N1
        erreurs_hidden = numpy.transpose(self.weights_ho)
        erreurs_hidden = numpy.dot(erreurs_hidden,erreurs)


        #Calcul du hidden gradient
        hidden_gradient = numpy.asmatrix(self.dsigmoid(numpy.asarray(calcul_ih)))
        hidden_gradient = numpy.multiply(hidden_gradient,erreurs_hidden)
        hidden_gradient = numpy.multiply(hidden_gradient,self.learning_rate)

        #Update des bias
        self.bias_ih = numpy.add(self.bias_ih,hidden_gradient)

        #Calcul des delta weights i-h
        tranposed_features = numpy.transpose(features)
        delta_weights_ih = numpy.multiply(hidden_gradient,tranposed_features)

        self.weights_ih = numpy.add(self.weights_ih,delta_weights_ih)


    def sigmoid(self,x):
        return 1 / (1 + numpy.exp(-x))

    #Pas vraiment le derivative de sigmoid puisque les outputs sont déjà sigmoid
    def dsigmoid(self,x):
        return x * (1 - x)
