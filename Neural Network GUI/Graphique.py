import pylab as plt
import numpy as np
class Graphique:

    def __init__(self,fig,img):
        self.ax1 = fig.add_subplot(211)
        self.ax1.axis("off")
        fig.set_figheight(10)
        fig.set_figwidth(10)
        self.pixels = np.array([img], dtype='float')
        self.pixels = self.pixels.reshape((28, 28))
        self.image = plt.imshow(self.pixels, cmap='gray')
        self.erreur_test = plt.text(-2.2,-0.60,"Nombre d'erreurs(sur ce test): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.bonne_test = plt.text(-2.2,-0.80,"Nombre de bonnes réponses (sur ce test): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.moy_test = plt.text(-2.2,-1,"Moyenne de bonnes réponses (sur ce test): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.erreur_total = plt.text(1.2,-0.60,"Nombre d'erreurs (totales): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.bonne_total = plt.text(1.2,-0.80,"Nombre de bonnes réponses (totales): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.moy_total = plt.text(1.2,-1,"Moyenne de bonnes réponses (totales): {}".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.nb_trained = plt.text(-0.2,-1.4,"Entrainer sur: {} nombres".format(0),transform=plt.gca().transAxes,fontsize=20)
        self.nb_guessed = plt.text(-2,0.4,"Prédiction: {}".format(0),transform=plt.gca().transAxes,fontsize=60)
        self.phase = plt.text(0.3,1.2,"Testing",transform=plt.gca().transAxes,fontsize=30)

        self.nb_e_t = 0
        self.nb_e_to = 0

        self.nb_b_t = 0
        self.nb_b_to = 0

        self.nb_moy_t = 0
        self.nb_moy_to = 0

    #Change l'image affichée
    def update_img(self,img):
        self.pixels = np.array([img], dtype='float')
        self.pixels = self.pixels.reshape((28, 28))
        self.image.set_data(self.pixels)
        plt.draw()


    #Change les stats affiché
    def update_stats(self,e_t,e_to,b_t,b_to,rep_donne,total_trained):
        self.nb_e_t += e_t
        self.nb_e_to += e_to

        self.nb_b_t += b_t
        self.nb_b_to += b_to

        self.nb_moy_t = np.round(self.nb_b_t / (self.nb_b_t + self.nb_e_t) * 100,2)
        self.nb_moy_to = np.round(self.nb_b_to / (self.nb_b_to + self.nb_e_to) * 100,2)


        self.erreur_test.set_text("Nombre d'erreurs(sur ce test): {}".format(self.nb_e_t))
        self.bonne_test.set_text("Nombre de bonnes réponses (sur ce test): {}".format(self.nb_b_t))
        self.moy_test.set_text("Moyenne de bonnes réponses (sur ce test): {}%".format(self.nb_moy_t))
        self.erreur_total.set_text("Nombre d'erreurs (totales): {}".format(self.nb_e_to))
        self.bonne_total.set_text("Nombre de bonnes réponses (totales): {}".format(self.nb_b_to))
        self.moy_total.set_text("Moyenne de bonnes réponses (totales): {}%".format(self.nb_moy_to))
        self.nb_guessed.set_text("Prédiction: {}".format(rep_donne))
        self.nb_trained.set_text("Entrainer sur: {} nombres".format(total_trained))
        plt.draw()
        plt.pause(0.005)
    #Clear les données test
    def clear_test(self):
        self.nb_e_t = 0
        self.nb_b_t = 0
        self.moy_t = 0

    #Commence training
    def show_train(self):
        self.phase.set_text("Training finished!")
        plt.draw()
        plt.pause(2)
        self.phase.set_text("Testing")
        plt.draw()
