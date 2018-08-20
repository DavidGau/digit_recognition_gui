from mnist import MNIST
import numpy as np
import time
import NeuralNetwork
import random
import Graphique
import pylab as plt





#Lorsque l'on souhaite train/guess une image, on la convertit d'abord en nombre entre 0 et 1
#Boucle qui prépare et envoi les donnés au training
def training():
    global total_training

    for i in range(total_training,total_training + 5000):
        image_a_transformer = image_training[i]
        image_transformer = [[pixel / 255] for pixel in image_a_transformer]

        label_a_transformer = label_training[i]
        array_label = [0,0,0,0,0,0,0,0,0,0]
        array_label[label_a_transformer] = 1
        array_label = [[element] for element in array_label]
        label_transformer = np.asmatrix(array_label)

        digit_network.train(image_transformer,label_transformer)
        total_training += 1


#Boucle qui prépare l'Envoi du test set
#Display égalment des statistiques

def calcul_stats(guess,label):
    global nb_erreur
    number_guessed = np.argmax(guess)
    erreur_bonne = (1,0,number_guessed)

    if number_guessed == label: #Si c'est un bon gself.ax2.plot(x,y)uess
        erreur_bonne = (0,1,number_guessed)
    else:
        nb_erreur += 1
    return erreur_bonne


def testing():
    global index_testing
    global total_training

    canvas.update_img(image_testing[index_testing])

    image_a_transformer = image_testing[index_testing]
    image_transformer = [[pixel / 255] for pixel in image_a_transformer]

    label = label_testing[index_testing]

    guess = np.round(digit_network.guess(image_transformer),2)

    reponse = calcul_stats(guess,label)

    canvas.update_stats(reponse[0],reponse[0],reponse[1],reponse[1],reponse[2],total_training)
    index_testing += 1

#Création du Neural Network
digit_network = NeuralNetwork.NeuralNetwork()


#Ouverture des donnée MNIST
mndata = MNIST("./")

image_training, label_training = mndata.load_training()
image_testing, label_testing = mndata.load_testing()

#randomize le training set
label_training = list(label_training)
randomize = list(zip(image_training,label_training))
random.shuffle(randomize)
image_training[:], label_training[:] = zip(*randomize)
print(len(label_training))
#randomize le test set
label_testing = list(label_testing)
randomize = list(zip(image_testing,label_testing))
random.shuffle(randomize)
image_testing[:], label_testing[:] = zip(*randomize)

index_testing = 0
nb_erreur = 0
total_training = 0
#Détection du clique sur le canvas
def onclick(event):
    global total_training

    if event.button == 1: #Clique pour guess
        global nb_erreur
        nb_erreur = 0
        canvas.clear_test()
        global index_testing
        for i in range(0,100):
            testing()
    elif event.button == 3: #Clique pour train
        training()
        canvas.show_train()
#Affichage graphique vite fait pour démontrer le fonctionnement du nn
fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', onclick)
canvas = Graphique.Graphique(fig,image_testing[0])
plt.show()






