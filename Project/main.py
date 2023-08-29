##############################################################################
# ------------------- PROJET ML: RESEAU DE NEURONES DIY -------------------- #
##############################################################################


################### IMPORTATION DES LIBRAIRIES ET MODULES ####################

import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
import seaborn as sns

from mltools import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE            # Import TSNE capability from scikit



########################### CLASSE ABSTRAITE LOSS ############################
##Classe abstraite pour le calcul du coût.
class Loss(object):

    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

##Classe pour la fonction de coût aux moindres carrés.
class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm( y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        return -2 * (y-yhat)

# Classe pour la fonction de coût cross-entropique.
class CE(Loss):
    def forward (self, y, yhat) :
        return - np.sum( y * yhat , axis = 1 )

    def backward (self, y, yhat) :
        return - y

#Classe pour la fonction de coût cross-entropique appliqué au log SoftMax.
class CESM(Loss):
    def forward (self, y, yhat) :
        return - np.sum( y * yhat , axis = 1 ) + np.log( np.sum( np.exp(yhat), axis = 1 ) )

    def backward (self, y, yhat) :
        s = SoftMax().forward( yhat )
        return - y + s * ( 1 - s )


class BCE (Loss) :
    def forward (self, y, yhat):
        return -(y* np.maximum(-100, np.log(yhat+0.01)) + (1-y) * np.maximum(-100, np.log(1-yhat+0.01)))

    def backward (self, y, yhat) :
        return - ((y/(yhat+0.01))- ((1-y)/(1-yhat+0.01)))

########################## CLASSE ABSTRAITE MODULE ###########################

## Classe abstraite représentant un module générique du réseau de neurones
class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        pass

    def forward(self, X):
        pass

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass

## Module représentant une couche linéaire
class Linear(Module):
    def __init__(self, input, output):
        self.input = input
        self.output = output
        self._parameters = 2 * ( np.random.rand(self.input, self.output) - 0.5 )
        self.zero_grad()

    def zero_grad(self):
        self._gradient = np.zeros((self.input, self.output))

    def forward(self, X):
        return np.dot( X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, X, delta):
        self._gradient += np.dot( X.T, delta )

    def backward_delta(self, X, delta):
        return np.dot( delta, self._parameters.T )


## Module représentant une couche de transformation tangeante hyperbolique
class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X)

    def backward_delta(self, X, delta):
        return ( 1 - np.tanh(X)**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass

## Module représentant une couche de transformation sigmoide
class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, X, delta):
        return ( np.exp(-X) / ( 1 + np.exp(-X) )**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass

##  Module représentant une couche de transformation SoftMax
class SoftMax (Module) :
    def __init__(self):
        super().__init__()

    def forward(self, X):
        e_x = np.exp(X)
        return e_x / e_x.sum( axis = 1 ).reshape(-1,1)

    def backward_delta(self, X, delta):
        s = self.forward( np.array(X) )
        return s * ( 1 - s ) * delta

    def update_parameters(self, gradient_step=1e-3):
        pass


############### NOTRE TOUT PREMIER RESEAU: REGRESSION LINEAIRE ###############

## Classe pour la régression linéaire
class RegLin:
    def fit(self, xtrain, ytrain, niter=100, gradient_step=1e-5):
        # Ajout d'un biais aux données
        #xtrain = add_bias(xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation de la couche et de la loss
        self.mse = MSELoss()
        self.linear = Linear(input, output)
        self.list_loss=[]
        for i in range(niter):

            # ETAPE 1: Calcul de l'état du réseau (phase forward)
            yhat = self.linear.forward(xtrain)

            # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)
            last_delta = self.mse.backward(ytrain, yhat)
            delta = self.linear.backward_delta(xtrain, last_delta)

            self.linear.backward_update_gradient(xtrain, delta)

            # ETAPE 3: Mise à jour des paramètres du réseau (matrice de poids w)
            self.linear.update_parameters(gradient_step)
            self.linear.zero_grad()
            self.list_loss.append(np.mean( self.mse.forward(ytrain, yhat) ))
        # Calcul de la loss
        self.last_loss = np.mean( self.mse.forward(ytrain, yhat) )

    def predict(self, xtest):
        return self.linear.forward(xtest)


############ NOTRE DEUXIEME RESEAU: CLASSIFICATION NON LINEAIRE ##############

##  Classe pour un classifieur non-linéaire
class NonLin:
    def fit(self, xtrain, ytrain, niter=100, gradient_step=1e-5, neuron=100):
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.mse = MSELoss()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()

        for i in range(niter):

            # ETAPE 1: Calcul de l'état du réseau (phase forward)
            res1 = self.linear_1.forward(xtrain)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.sigmoide.forward(res3)

            # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)
            last_delta = self.mse.backward(ytrain, res4)

            delta_sig = self.sigmoide.backward_delta(res3, last_delta)
            delta_lin = self.linear_2.backward_delta(res2, delta_sig)
            delta_tan = self.tanh.backward_delta(res1, delta_lin)

            self.linear_1.backward_update_gradient(xtrain, delta_tan)
            self.linear_2.backward_update_gradient(res2, delta_sig)

            # ETAPE 3: Mise à jour des paramètres du réseau (matrice de poids w)
            self.linear_1.update_parameters(gradient_step)
            self.linear_2.update_parameters(gradient_step)
            self.linear_1.zero_grad()
            self.linear_2.zero_grad()

        # Affichage de la loss
        print("\nErreur mse :", np.mean( self.mse.forward(ytrain, res4) ) )

    def predict(self, xtest):
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        res = self.linear_1.forward(xtest)
        res = self.tanh.forward(res)
        res = self.linear_2.forward(res)
        res = self.sigmoide.forward(res)

        return np.argmax(res, axis = 1)


######################## CLASSE SEQUENTIEL ET OPTIM ##########################

## Factorisation des réseaux de neurones

class Sequentiel:
    def __init__(self, modules, loss):
        self.modules = modules
        self.loss = loss

    def fit(self, xtrain, ytrain):
        # ETAPE 1: Calcul de l'état du réseau (phase forward)
        res_forward = [ self.modules[0].forward(xtrain) ]

        for j in range(1, len(self.modules)):
            res_forward.append( self.modules[j].forward( res_forward[-1] ) )

        # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
        #          par rapport aux paramètres et aux entrées)
        deltas =  [ self.loss.backward( ytrain, res_forward[-1] ) ]

        for j in range(len(self.modules) - 1, 0, -1):
            deltas += [ self.modules[j].backward_delta( res_forward[j-1], deltas[-1] ) ]

        return res_forward, deltas


class Optim:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.sequentiel = Sequentiel(net, loss)

    def step(self, batch_x, batch_y):
        # ETAPE 1: Calcul de l'état du réseau (phase forward) et passe backward
        res_forward, deltas = self.sequentiel.fit(batch_x, batch_y)

        # ETAPE 2: Phase backward par rapport aux paramètres et mise-à-jour
        for j in range(len(self.net)):

            # Mise-à-jour du gradient
            if j == 0:
                self.net[j].backward_update_gradient(batch_x, deltas[-1])
            else:
                self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

            # Mise-à-jour des paramètres
            self.net[j].update_parameters(self.eps)
            self.net[j].zero_grad()

    def predict(self, xtest):
        # Phase passe forward
        res_forward = [ self.net[0].forward(xtest) ]

        for j in range(1, len(self.net)):
            res_forward.append( self.net[j].forward( res_forward[-1] ) )

        return np.argmax(res_forward[-1], axis=1)


###### NOTRE TROISIEME RESEAU: CLASSIFICATION NON-LINEAIRE SEQUENTIELLE ######

## Classe pour un classifieur non-linéaire version séquentielle.
class NonLin2:
    ##La fonction fit correspond à la classe SGD demandée
    def fit(self, xtrain, ytrain, batch_size=1, neuron=10, niter=1000, gradient_step=1e-5):
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.mse = CESM()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()

        # Liste des couches du réseau de neurones
        modules = [ self.linear_1, self.tanh, self.linear_2, self.sigmoide ]

        # Apprentissage du réseau de neurones
        self.optim  = Optim(modules, self.mse, gradient_step)

        for i in range(niter):
            # Tirage d'un batch de taille batch_size et mise-à-jour
            inds = [ rd.randint(0, len(xtrain) - 1) for j in range(batch_size) ]
            self.optim.step( xtrain[inds], ytrain[inds] )


    def SGD(self, net, loss, xtrain, ytrain, batch_size=1, niter=1000, gradient_step=1e-5):
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Apprentissage du réseau de neurones
        self.optim  = Optim(net, loss, gradient_step)

        # Liste de variables pour simplifier la création des batchs
        card = xtrain.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)

        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in range(niter):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            self.optim.step(xtrain[batch], ytrain[batch])



    def predict(self, xtest):
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        return self.optim.predict(xtest)


################## NOTRE CINQUIÈME RESEAU: AUTO-ENCODEUR ##################
class AutoEncodeur :
    def codage (self, xtrain, modules):
        # ETAPE 1: Calcul de l'état du réseau (phase forward)
        res_forward = [ modules[0].forward(xtrain) ]

        for j in range(1, len(modules)):
            #print("AFFICHAGE",type(modules[j]).__name__,res_forward[-1])

            res_forward.append( modules[j].forward( res_forward[-1] ) )

        return res_forward

    def fit(self, xtrain,batch_size=1, neuron=10, niter=1000, gradient_step=1e-5, output = 10):
        # Ajout d'un biais aux données
        #xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.bce = BCE()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()
        self.linear_3 = Linear (output, neuron)
        self.linear_4 = Linear (neuron, input)


        # Liste des couches du réseau de neurones
        self.modules_enco = [ self.linear_1, self.tanh, self.linear_2, self.tanh ]
        self.modules_deco = [ self.linear_3, self.tanh, self.linear_4, self.sigmoide ]
        self.net = self.modules_enco + self.modules_deco

        for i in range(niter):
            res_forward_enco = self.codage(xtrain, self.modules_enco)
            res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
            res_forward = res_forward_enco + res_forward_deco
            #print("SIG", res_forward[-1])
            #print("RES FORWARD:",res_forward)
            if(i%100==0):
                print(np.sum(self.bce.forward(xtrain, res_forward[-1]), axis=1))

            # Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)

            deltas =  [ self.bce.backward( xtrain, res_forward[-1] ) ]

            for j in range(len(self.net) - 1, 0, -1):
                deltas += [self.net[j].backward_delta( res_forward[j-1], deltas[-1] ) ]


            #Phase backward par rapport aux paramètres et mise-à-jour
            for j in range(len(self.net)):
                # Mise-à-jour du gradient
                if j == 0:
                    self.net[j].backward_update_gradient(xtrain, deltas[-1])
                else:
                    self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

                # Mise-à-jour des paramètres
                self.net[j].update_parameters(gradient_step)
                self.net[j].zero_grad()

    def predict (self, xtest) :
        #xtest = add_bias(xtest)
        res_forward_enco = self.codage(xtest, self.modules_enco)
        res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
        return res_forward_enco[-1], res_forward_deco[-1]


##############################################################################
# ------------------------------ UTILITAIRES ------------------------------- #
##############################################################################

## Fonction permettant d'ajouter un biais aux données.
def add_bias(datax):
    bias = np.ones((len(datax), 1))
    return np.hstack((bias, datax))


######################## CHARGEMENT DES DONNEES USPS #########################

## Chargement des données
def load_usps(filename):
    with open(filename, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


def get_usps(l, datax, datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy

    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    datax_new, datay_new = np.vstack(tmp[0]),np.hstack(tmp[1])

    return datax_new, datay_new


def show_usps(datax):
    plt.imshow(datax.reshape((16,16)),interpolation="nearest",cmap="magma")


def plot(datax, datay, model, name=''):
    plot_frontiere(datax,lambda x : model.predict(x),step=100)
    plot_data(datax,datay.reshape(1,-1)[0])
    plt.title(name)
    plt.show()

def load_data(classes=10):


    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"

    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)

    # Création des données d'apprentissage et des données d'entraînement
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)

    #bruit
    bruit = np.random.rand(xtest.shape[0], xtest.shape[1])
    #xtrain = np.where(xtrain+bruit <= 2, xtrain+bruit, xtrain)
    xtest = np.where(xtest+bruit <= 2, xtest+bruit, xtest)

    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)

    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot

    return xtrain, ytrain, xtest, ytest

##############################################################################
# ----------------------------- FONCTION TEST------------------------------- #
##############################################################################

def mainLineaire_reg(a=97.28, sigma=50):
    # On décide arbitrairement d'un coefficient directeur a
    print('\nCoefficient directeur réel:', a)

    # Génération des données d'entraînement
    xtrain = np.array( [ x for x in np.linspace(0, 2.0, 100) ] ).reshape(-1,1)
    ytrain = np.array( [ a * x + rd.uniform(-sigma,sigma) for x in np.linspace(0,2.0,100) ] )

    # Création de notre modèle de régression linéaire
    rl = RegLin()

    # Phase d'entraînement puis prédiction des classes des données de xtrain
    rl.fit(xtrain, ytrain.reshape(-1,1), niter=500, gradient_step=1e-5)

    w = rl.linear._parameters[0][0]
    print('Coefficient linéaire prédit:', w)

    # Affichage de la loss
    print("\nErreur mse :", rl.last_loss )

    # Affichage des données et de la droite prédite
    toPlot = [ w * x[0] for x in xtrain ]
    plt.figure()
    plt.title('Régression linéaire')
    plt.scatter(xtrain.reshape(1,-1), ytrain, s = 1, c = 'midnightblue', label='data')
    plt.plot(xtrain.reshape(1,-1)[0], toPlot, color = 'mediumslateblue', label='model')
    plt.legend()

    plt.figure()
    plt.title('Evolution de la MSE')
    plt.plot(rl.list_loss, label='loss')
    plt.legend()

def mainLineaire_classif(eps=0.1):

    ## NONLIN2 SUR LES DONNEES GEN_ARTI


    # Création de données artificielles suivant 2 gaussiennes
    datax, datay = gen_arti(epsilon=eps, data_type=0)

    # Normalisation des données
    scaler = StandardScaler()
    datax = scaler.fit_transform(datax)

    # One-Hot Encoding
    datay = np.array([ 0 if d == -1 else 1 for d in datay ])
    onehot = np.zeros((datay.size, 2))
    onehot[ np.arange(datay.size), datay ] = 1
    datay = onehot

    # Création et test sur un réseau de neurones non linéaire

    time_start = time.time()

    batch, output = datay.shape
    batch, input = datax.shape

    neuron = 4

    mse = CESM()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    sigmoide = Sigmoide()

    net = [linear_1, tanh, linear_2, sigmoide]

    nonlin = NonLin2()
    nonlin.SGD(net, mse, datax, datay, batch_size=len(datay), niter=2000, gradient_step=0.001)
    #nonlin.fit(datax, datay, batch_size=len(datay), niter=1000, neuron=10, gradient_step=1e-3)

    # Test sur les données d'apprentissage
    ypred = nonlin.predict(datax)
    datay = np.argmax(datay, axis=1)

    print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == datay ))
    plot(datax, datay, nonlin)


def data_4_gauss ():
    # Création de données artificielles suivant 4 gaussiennes
    datax, datay = gen_arti(epsilon=0.1, data_type=1)

    # Normalisation des données
    scaler = StandardScaler()
    datax = scaler.fit_transform(datax)

    # One-Hot Encoding
    datay = np.array([ 0 if d == -1 else 1 for d in datay ])
    onehot = np.zeros((datay.size, 2))
    onehot[ np.arange(datay.size), datay ] = 1
    datay = onehot

    return datax, datay

def mainNonLineaire(datax, datay, neuron, max_iter, size, step=0.001):

    ## NONLIN2 SUR LES DONNEES GEN_ARTI


    # Création et test sur un réseau de neurones non linéaire

    time_start = time.time()

    batch, output = datay.shape
    batch, input = datax.shape

    mse = MSELoss()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    sigmoide = Sigmoide()

    net = [linear_1, tanh, linear_2, sigmoide]

    nonlin = NonLin2()
    nonlin.SGD(net, mse, datax, datay, batch_size=size, niter=max_iter, gradient_step=step)
    #nonlin.fit(datax, datay, batch_size=len(datay), niter=1000, neuron=10, gradient_step=1e-3)

    # Test sur les données d'apprentissage
    ypred = nonlin.predict(datax)
    datay = np.argmax(datay, axis=1)

    #print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == datay ))
    #plot(datax, datay, nonlin)


##############################################################################
# ------------------------- TESTS SUR LES DONNEES -------------------------- #
##############################################################################

if (__name__ == "__main__"):
	"""
    ## Test regression linéaire
    #mainLineaire_reg(a=10, sigma=6)
    ## Test classification linéaire
    #mainLineaire_classif(eps=1)

    ## NONLIN ET NONLIN2 SUR LES DONNEES USPS
    datax, datay = data_4_gauss ()
    step = [0.0005, 0.001, 0.01, 0.05, 0.1, 0.2]
    batch = [1, 200, 500, 800, 1000]
    for i in [0.1]:
        for j in batch:
            for k in [10,40,120]:
                    print(j)
                    print(k)
                    mainNonLineaire(datax, datay, neuron=3, max_iter=k, size=j, step=i)
                    if(j==1001):
                        mainNonLineaire(datax, datay, neuron=3, max_iter=k, size=len(datay), step=i)
                    print("\n")




	# Création et test sur un réseau de neurones non linéaire

	time_start = time.time()
	# pour mse
	#nonlin = NonLin()
	#nonlin.fit(xtrain, ytrain, niter=100, neuron=100, gradient_step=0.01)
	#pour cesm
	#nonlin = NonLin2()
	#nonlin.fit(xtrain, ytrain, niter=50000, neuron=300, gradient_step=0.01)
	# pour auto-encodeur
	auto = AutoEncodeur()
	auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4)
	#ytrain = np.argmax(ytrain, axis=1)

	# Test sur les données d'apprentissage
	y_enco,y_deco= auto.predict(xtest)

	kmeans = KMeans(n_clusters = 10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
	pred_enco = kmeans.fit_predict(y_enco)
	pred_deco = kmeans.fit_predict(y_deco)

	plt.figure()
	show_usps(xtest[1])
	plt.show()
	plt.figure()
	show_usps(y_deco[1])
	plt.show()

	#t-sne
	perplexity_list = [5, 10, 30, 50, 100]
	tsne_perp = [ TSNE(n_components=2, random_state=0, perplexity=perp) for perp in perplexity_list]
	tsne_perp_data = [ tsne.fit_transform(y_deco) for tsne in tsne_perp]
	plt.figure(figsize=(15,15))
	for i in range(len(perplexity_list)):
	    plt.subplot(3,2, i+1)
	    plt.title("Perplexity " + str(perplexity_list[i]))
	    plt.scatter(tsne_perp_data[i][:,0],tsne_perp_data[i][:,1], c=pred_deco)
	plt.subplot(3,2, i+2)
	plt.axis('off')

	print("\nTemps d'exécution: ", time.time() - time_start )
	#print("Score de bonne classification: ", np.mean( ypred == ytest ))
	"""

	"""
	# rappel, précision (mse vs ce + softmax) ====================================================================

	# Chargement des données USPS
	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)
	alltestx,alltesty = load_usps(uspsdatatest)

	# Création des données d'apprentissage et des données d'entraînement
	classes = 10
	xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
	xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)

	# Normalisation
	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)
	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)

	# One-Hot Encoding
	onehot = np.zeros((ytrain.size,classes))
	onehot[ np.arange(ytrain.size), ytrain ] = 1
	ytrain = onehot


	# Récupération des tailles des entrées
	batch, output = ytrain.shape
	batch, input = xtrain.shape

	sigmoide = Sigmoide()
	softmax = SoftMax()
	mse = MSELoss()
	ce = CESM()
	tanh = TanH()

	recalls_CESM = []
	recalls_mse = []

	precisions_CESM = []
	precisions_mse = []

	neurons = []

	digit = 3

	l = 5

	for neuron in range(1,30,1):
		#recall_mse = 0
		#recall_cesm = 0
		precision_mse = 0
		precision_cesm = 0

		print(neuron)

		division_par_zero = False



		for i in range(l):



			# Initialisation des couches du réseau et de la loss
			linear_1 = Linear(input+1, neuron)
			linear_2 = Linear(neuron, output)


			# Liste des couches du réseau de neurones
			net = [linear_1, tanh, linear_2, sigmoide]

			# Création et test sur un réseau de neurones non linéaire

			#time_start = time.time()

			nonlin = NonLin2()
			nonlin.SGD(net, mse, xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)
			#nonlin.fit(xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)

			# Test sur les données d'apprentissage
			ypred = nonlin.predict(xtest)


			# taux de bonne classification
			#rate = np.mean( ypred == ytest )
			TP = (ypred[ypred == ytest] == digit).sum()
			TP_plus_FP = (ypred == digit).sum()



			if(TP_plus_FP == 0):
				division_par_zero = True
				break

			precision_mse += TP/TP_plus_FP

			#P = (ytest == digit).sum()
			#recall_mse += TP/P



			nonlin = NonLin2()
			nonlin.SGD(net, ce, xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)
			#nonlin.fit(xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)

			# Test sur les données d'apprentissage
			ypred = nonlin.predict(xtest)

			# taux de bonne classification
			# rate = np.mean( ypred == ytest )
			TP = (ypred[ypred == ytest] == digit).sum()
			TP_plus_FP = (ypred == digit).sum()
			precision_cesm += TP/TP_plus_FP

			#P = (ytest == digit).sum()
			#recall_cesm += TP/P

		#recalls_mse.append(rate_mse/l)
		#recalls_CESM.append(rate_cesm/l)
		if not division_par_zero:
			neurons.append(neuron)
			precisions_mse.append(precision_mse/l)
			precisions_CESM.append(precision_cesm/l)


	#plt.plot(neurons,recalls_CESM,label="CE + Softmax")
	#plt.plot(neurons,recalls_mse,label="MSE")

	plt.plot(neurons,precisions_CESM,label="CE + Softmax")
	plt.plot(neurons,precisions_mse,label="MSE")

	plt.xlabel("Nombre de neurones par couche")
	#plt.ylabel("Taux de bonne classification")
	#plt.ylabel("Rappel (TP/P)")
	plt.ylabel("Précision : TP/(TP+FP)")
	plt.legend()
	plt.show()
	"""

	# test de reconstructions d'images ===========================================================================================================
	"""

	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)


	auto = AutoEncodeur()

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)



	auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4)
	#ytrain = np.argmax(ytrain, axis=1)

	# Test sur les données d'apprentissage
	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)

	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)

	y_enco,y_deco= auto.predict(xtest)

	fig=plt.figure(figsize=(4,5))

	# 0
	fig.add_subplot(4,5,1)
	plt.imshow(xtest[5].reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,2)
	plt.imshow(xtest[359].reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,3)
	plt.imshow(xtest[623].reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,4)
	plt.imshow(xtest[821].reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,5)
	plt.imshow(xtest[987].reshape(16,16))
	plt.axis('off')



	# 0
	fig.add_subplot(4,5,6)
	plt.imshow(y_deco[5].reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,7)
	plt.imshow(y_deco[359].reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,8)
	plt.imshow(y_deco[623].reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,9)
	plt.imshow(y_deco[821].reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,10)
	plt.imshow(y_deco[987].reshape(16,16))
	plt.axis('off')


	# 5
	fig.add_subplot(4,5,11)
	plt.imshow(xtest[1189].reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,12)
	plt.imshow(xtest[1347].reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,13)
	plt.imshow(xtest[1517].reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,14)
	plt.imshow(xtest[1665].reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,15)
	plt.imshow(xtest[1830].reshape(16,16))
	plt.axis('off')



	# 5
	fig.add_subplot(4,5,16)
	plt.imshow(y_deco[1189].reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,17)
	plt.imshow(y_deco[1347].reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,18)
	plt.imshow(y_deco[1517].reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,19)
	plt.imshow(y_deco[1665].reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,20)
	plt.imshow(y_deco[1830].reshape(16,16))
	plt.axis('off')

	plt.show()
	"""
	"""
	# =================================================================================================================
	# moyenne des images VS moyenne des reconstructions ===============================================================
	"""
	"""
	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)

	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)

	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)


	#for latent_space_size in range(10,100,10):
	auto = AutoEncodeur()
	auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4)
	#ytrain = np.argmax(ytrain, axis=1)



	# Test sur les données d'apprentissage
	y_enco_0,y_deco_0= auto.predict(xtest[ytest==0])
	y_enco_1,y_deco_1= auto.predict(xtest[ytest==1])
	y_enco_2,y_deco_2= auto.predict(xtest[ytest==2])
	y_enco_3,y_deco_3= auto.predict(xtest[ytest==3])
	y_enco_4,y_deco_4= auto.predict(xtest[ytest==4])
	y_enco_5,y_deco_5= auto.predict(xtest[ytest==5])
	y_enco_6,y_deco_6= auto.predict(xtest[ytest==6])
	y_enco_7,y_deco_7= auto.predict(xtest[ytest==7])
	y_enco_8,y_deco_8= auto.predict(xtest[ytest==8])
	y_enco_9,y_deco_9= auto.predict(xtest[ytest==9])

	fig=plt.figure(figsize=(4,5))

	# 0
	fig.add_subplot(4,5,1)
	plt.imshow(np.mean(xtest[ytest==0],axis=0).reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,2)
	plt.imshow(np.mean(xtest[ytest==1],axis=0).reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,3)
	plt.imshow(np.mean(xtest[ytest==2],axis=0).reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,4)
	plt.imshow(np.mean(xtest[ytest==3],axis=0).reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,5)
	plt.imshow(np.mean(xtest[ytest==4],axis=0).reshape(16,16))
	plt.axis('off')



	# 0
	fig.add_subplot(4,5,6)
	plt.imshow(np.mean(y_deco_0,axis=0).reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,7)
	plt.imshow(np.mean(y_deco_1,axis=0).reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,8)
	plt.imshow(np.mean(y_deco_2,axis=0).reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,9)
	plt.imshow(np.mean(y_deco_3,axis=0).reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,10)
	plt.imshow(np.mean(y_deco_4,axis=0).reshape(16,16))
	plt.axis('off')


	# 5
	fig.add_subplot(4,5,11)
	plt.imshow(np.mean(xtest[ytest==5],axis=0).reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,12)
	plt.imshow(np.mean(xtest[ytest==6],axis=0).reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,13)
	plt.imshow(np.mean(xtest[ytest==7],axis=0).reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,14)
	plt.imshow(np.mean(xtest[ytest==8],axis=0).reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,15)
	plt.imshow(np.mean(xtest[ytest==9],axis=0).reshape(16,16))
	plt.axis('off')



	# 5
	fig.add_subplot(4,5,16)
	plt.imshow(np.mean(y_deco_5,axis=0).reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,17)
	plt.imshow(np.mean(y_deco_6,axis=0).reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,18)
	plt.imshow(np.mean(y_deco_7,axis=0).reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,19)
	plt.imshow(np.mean(y_deco_8,axis=0).reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,20)
	plt.imshow(np.mean(y_deco_9,axis=0).reshape(16,16))
	plt.axis('off')

	plt.show()

	"""
	# moyenne des distances intra-clusters =============================================================
	"""
	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)

	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)

	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)

	_,input_ = xtrain.shape


	auto = AutoEncodeur()
	print(input_)
	auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4,output=10)
	#ytrain = np.argmax(ytrain, axis=1)

	y_enco,y_deco= auto.predict(xtest)

	kmeans = KMeans(n_clusters = 10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)

	pred_enco = kmeans.fit_predict(y_enco)



	tsne = TSNE(n_components=2)
	tsne_data = tsne.fit_transform(y_enco)

	df_subset = {}

	df_subset['dimension 1'] = tsne_data[:,0]
	df_subset['dimension 2'] = tsne_data[:,1]

	plt.figure(figsize=(8,5))
	sns.scatterplot(
    x="dimension 1", y="dimension 2",
    hue=pred_enco,
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)

	plt.show()
	"""
	# ajout du bruit ===================================================================================
	"""
	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)

	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)





	epsilon = np.random.normal(0,1.5,size=xtest.shape)

	noised_xtest = xtest + epsilon

	scaler = MinMaxScaler()
	xtest = scaler.fit_transform(xtest)



	scaler = StandardScaler()
	noised_xtest = scaler.fit_transform(noised_xtest)
	maLoss = BCE()

	errors = []
	latent_space_sizes = []

	_,input_ = xtrain.shape

	for latent_space_size in range(1,input_,25):
		print("latent_space_size = ",latent_space_size)
		#for latent_space_size in range(10,100,10):
		auto = AutoEncodeur()

		auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4,output=latent_space_size)
		#ytrain = np.argmax(ytrain, axis=1)

		# Test sur les données d'apprentissage
		y_enco,y_deco= auto.predict(noised_xtest)

		scaler = MinMaxScaler()
		y_deco = scaler.fit_transform(y_deco)
		error = np.sum(np.mean(maLoss.forward(xtest, y_deco), axis=1))

		errors.append(error)
		latent_space_sizes.append(latent_space_size)

	plt.plot(latent_space_sizes,errors,marker='s')
	plt.xlabel("Taille de l'espace latent")
	plt.ylabel("BCE")
	plt.legend()
	plt.show()
	"""
	# affichage d'images bruités, débruités ==========================================================================
	"""
	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)

	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)





	epsilon = np.random.normal(0,0.6,size=xtest.shape)

	noised_xtest = xtest + epsilon

	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)



	scaler = StandardScaler()
	noised_xtest = scaler.fit_transform(noised_xtest)
	maLoss = BCE()

	errors = []
	latent_space_sizes = []

	_,input_ = xtrain.shape



	#for latent_space_size in range(10,100,10):
	auto = AutoEncodeur()

	auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4,output=3)
	#ytrain = np.argmax(ytrain, axis=1)

	# Test sur les données d'apprentissage
	y_enco,y_deco= auto.predict(noised_xtest)

	#scaler = MinMaxScaler()
	#y_deco = scaler.fit_transform(y_deco)



	fig=plt.figure(figsize=(4,5))

	# 0
	fig.add_subplot(4,5,1)
	plt.imshow(noised_xtest[5].reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,2)
	plt.imshow(noised_xtest[359].reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,3)
	plt.imshow(noised_xtest[623].reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,4)
	plt.imshow(noised_xtest[821].reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,5)
	plt.imshow(noised_xtest[987].reshape(16,16))
	plt.axis('off')



	# 0
	fig.add_subplot(4,5,6)
	plt.imshow(y_deco[5].reshape(16,16))
	plt.axis('off')

	# 1
	fig.add_subplot(4,5,7)
	plt.imshow(y_deco[359].reshape(16,16))
	plt.axis('off')

	# 2
	fig.add_subplot(4,5,8)
	plt.imshow(y_deco[623].reshape(16,16))
	plt.axis('off')

	# 3
	fig.add_subplot(4,5,9)
	plt.imshow(y_deco[821].reshape(16,16))
	plt.axis('off')

	# 4
	fig.add_subplot(4,5,10)
	plt.imshow(y_deco[987].reshape(16,16))
	plt.axis('off')


	# 5
	fig.add_subplot(4,5,11)
	plt.imshow(noised_xtest[1187].reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,12)
	plt.imshow(noised_xtest[1347].reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,13)
	plt.imshow(noised_xtest[1517].reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,14)
	plt.imshow(noised_xtest[1664].reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,15)
	plt.imshow(noised_xtest[1830].reshape(16,16))
	plt.axis('off')



	# 5
	fig.add_subplot(4,5,16)
	plt.imshow(y_deco[1187].reshape(16,16))
	plt.axis('off')

	# 6
	fig.add_subplot(4,5,17)
	plt.imshow(y_deco[1347].reshape(16,16))
	plt.axis('off')

	# 7
	fig.add_subplot(4,5,18)
	plt.imshow(y_deco[1517].reshape(16,16))
	plt.axis('off')

	# 8
	fig.add_subplot(4,5,19)
	plt.imshow(y_deco[1664].reshape(16,16))
	plt.axis('off')

	# 9
	fig.add_subplot(4,5,20)
	plt.imshow(y_deco[1830].reshape(16,16))
	plt.axis('off')

	plt.show()
	"""
	# ========================================================================


	uspsdatatrain = "data/USPS_train.txt"
	uspsdatatest = "data/USPS_test.txt"

	alltrainx,alltrainy = load_usps(uspsdatatrain)

	xtrain, ytrain = get_usps([i for i in range(10)], alltrainx, alltrainy)

	scaler = StandardScaler()
	xtrain = scaler.fit_transform(xtrain)

	alltestx,alltesty = load_usps(uspsdatatest)
	xtest, ytest = get_usps([i for i in range(10)], alltestx, alltesty)

	scaler = StandardScaler()
	xtest = scaler.fit_transform(xtest)

	_,input_ = xtrain.shape

	latent_space_sizes = []
	purities = []

	for latent_space_size in range(1,100,9):
		print("latent_space_size = ",latent_space_size)
		auto = AutoEncodeur()
		auto.fit(xtrain, niter=500, neuron=100, gradient_step=1e-4,output=10)
		#ytrain = np.argmax(ytrain, axis=1)

		y_enco,y_deco= auto.predict(xtest)

		kmeans = KMeans(n_clusters = 10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)

		pred_deco = kmeans.fit_predict(y_deco)

		purity = 0

		# pour chaque cluster
		for i in range(10):

			max_ = 0

			# on cherche la classe majoritaire
			for j in range(10):
				nb_classe_j = 0

				# on compte le nb d'exemples possédant la classe j dans le cluster i
				for k in range(len(pred_deco)):
					if(pred_deco[k] == i and ytest[k] == j):
						nb_classe_j += 1

				if(nb_classe_j > max_):
					max_ = nb_classe_j
					classe_majoritaire = j

			purity += max_



		purities.append(purity/len(ytest))
		latent_space_sizes.append(latent_space_size)


	plt.plot(latent_space_sizes,purities,marker='s')
	plt.ylabel("Pureté moyenne")
	plt.xlabel("Taille de l'espace latent")
	plt.legend()
	plt.show()










	"""
	"""


	"""
	"""

		#print("\nTemps d'exécution: ", time.time() - time_start )
		#print("Score de bonne classification: ", np.mean( ypred == ytest ))

	"""
	## NONLIN2 SUR LES DONNEES GEN_ARTI


	# Création de données artificielles suivant 4 gaussiennes
	datax, datay = gen_arti(epsilon=0.1, data_type=1)

	# Normalisation des données
	scaler = StandardScaler()
	datax = scaler.fit_transform(datax)

	# One-Hot Encoding
	datay = np.array([ 0 if d == -1 else 1 for d in datay ])
	onehot = np.zeros((datay.size, 2))
	onehot[ np.arange(datay.size), datay ] = 1
	datay = onehot

	# Création et test sur un réseau de neurones non linéaire

	time_start = time.time()

	nonlin = NonLin2()
	nonlin.fit(datax, datay, batch_size=len(datay), niter=100, neuron=3, gradient_step=1e-3)

	# Test sur les données d'apprentissage
	ypred = nonlin.predict(datax)
	datay = np.argmax(datay, axis=1)

	print("\nTemps d'exécution: ", time.time() - time_start )
	print("Score de bonne classification: ", np.mean( ypred == datay ))
	plot (datax, datay, nonlin)
	"""