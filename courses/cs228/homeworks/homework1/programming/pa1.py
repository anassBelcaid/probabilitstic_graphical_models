"""
CS 228: Probabilistic Graphical Models
Winter 2018
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover, Luis Perez
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
from random import choice
from scipy.special import logsumexp



NUM_PIXELS = 28*28


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency',
                   savefile='hist'):
  '''
  Plots a histogram.
  '''

  plt.figure()
  plt.hist(data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.savefig(savefile, bbox_inches='tight')
  plt.show()
  plt.close()

  return


def get_p_z1(z1_val):
  '''
  Helper. Computes the prior probability for variable z1 to take value z1_val.
  P(Z1=z1_val)
  '''

  return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
  '''
  Helper. Computes the prior probability for variable z2 to take value z2_val.
  P(Z2=z2_val)
  '''

  return bayes_net['prior_z2'][z2_val]


def get_p_xk_cond_z1_z2(z1_val, z2_val, k):
  '''
  Helper. Computes the conditional probability that variable xk assumes value 1
  given that z1 assumes value z1_val and z2 assumes value z2_val
  P(Xk = 1 | Z1=z1_val , Z2=z2_val)
  '''

  return bayes_net['cond_likelihood'][(z1_val, z2_val)][0, k-1]


def get_p_x_cond_z1_z2(z1_val, z2_val):
  '''
  Computes the conditional probability of the entire vector x for x = 1,
  given that z1 assumes value z1_val and z2 assumes value z2_val
  '''
  return np.array([ get_p_xk_cond_z1_z2(z1_val, z2_val,k) for k in
      range(NUM_PIXELS)])


def get_network_conditionals():
    pass


def get_pixels_sampled_from_p_x_joint_z1_z2():
  '''
  This function should sample from the joint probability distribution specified
  by the model, and return the sampled values of all the pixel variables (x).
  Note that this function should return the sampled values of ONLY the pixel
  variables (x), discarding the z part.
  '''
  
  #sampling from z1
  z1 = choice(disc_z1)

  #sampling from z2
  z2 = np.random.choice(disc_z2)

  #conditional probabilities
  cond_x = get_p_x_cond_z1_z2(z1, z2)

  return np.where(cond_x > 0.5, 1, 0)
  # return cond_x
  

def q4():
  '''
  Plots the pixel variables sampled from the joint distribution as 28 x 28
  images. Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2
  '''

  plt.figure()
  for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2(
    ).reshape(28, 28), cmap='gray')
    plt.title('Sample: ' + str(i+1))
  plt.tight_layout()
  plt.savefig('a4', bbox_inches='tight')
  plt.show()
  plt.close()

  return


def q5():
  '''
  Plots the expected images for each latent configuration on a 2D grid.
  Your job is to implement get_p_x_cond_z1_z2
  '''

  canvas = np.empty((28*len(disc_z1), 28*len(disc_z2)))
  for i, z1_val in enumerate(disc_z1):
    for j, z2_val in enumerate(disc_z2):
      canvas[(len(disc_z1)-i-1)*28:(len(disc_z2)-i)*28, j*28:(j+1)*28] = \
          get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

  plt.figure()
  plt.imshow(canvas, cmap='gray')
  plt.tight_layout()
  plt.savefig('a5', bbox_inches='tight')
  plt.show()
  plt.close()

  return

def get_probabilities():
    """
    get the priors for Z as a flattened array
    """

    #priors on Z
    priors = np.zeros((25, 25))


    #conditional on z
    Px = np.zeros((25*25, NUM_PIXELS))

    for i, z1 in enumerate(disc_z1):
        for j, z2 in enumerate(disc_z2):
            priors[i,j] = get_p_z1(z1) * get_p_z2(z2)

            #comppute conditional
            Px[np.ravel_multi_index((i,j), (25,25))] =\
                    get_p_x_cond_z1_z2(z1, z2)

    return priors.flatten(), Px, 1- Px


def marginal_log_likelihood(dataSet):
    """return the marginal log_likelihood of the image
    \sum_z1 \sum_z2 p(z1, z2, X)

    :X: The observed Images (image by row)
    :returns: log likelihood
    """

    z_shape = (25, 25)          # two dimensional shape for prior
    Z_prior, P0, P1 = get_probabilities()


    # going to the logarithmic space
    Z_prior = np.log(Z_prior)
    P0      = np.log(P0)
    P1      = np.log(P1)


    logs = np.zeros(len(dataSet))

    #getting the values
    i=0
    for X in dataSet:
        #probabilities 
        V = np.where(X == 0, P0, P1)

        #adding prior
        V = np.c_[Z_prior, V]

        #considering logsumexp
        V =  np.sum(V, axis = 1)
        V = logsumexp(V)

        logs[i] = V
        i +=  1
    return logs

def get_validation_stats():
  """
  Function to get the validation statistics
  """
  mat = loadmat('q6.mat')
  validation = mat['val_x']

  #get likelikhood
  # logs = marginal_log_likelihood(validation)
  # with open("logs","wb") as F:
  #     pkl.dump(logs, F)

  with open("logs", "rb") as F:
    logs = pkl.load(F)


  return np.mean(logs), np.std(logs)



def q6():
  '''
  Loads the data and plots the histograms.
  '''

  mat = loadmat('q6.mat')
  validation = mat['val_x']
  test       = mat['test_x']


  #getting the validation statistics
  mean, Std = get_validation_stats()

  #geting the marginal likelihood on the test
  logs_test = marginal_log_likelihood(test)

  #classification
  mask  = np.abs(logs_test - mean) <= 3*Std


  #saving the histograms
  plot_histogram(logs_test[mask],"positive",savefile="positive.png")
  plot_histogram(logs_test[np.logical_not( mask)],"negative",savefile="negative.png")


def conditional_prob_on_I(Img):
  """
  Return the conditional probabilities on Z1,Z2(flattend)
  condtionned on the Img
  """

  prior, P0, P1 = get_probabilities()

  #combine Images on P0 and P1
  PI = np.where(Img == 0, P0, P1)

  #noramlie PI
  PI = np.c_[prior, PI]


  #compute the expectation

  Z = disc_z1
  z1 = np.repeat(Z,25).reshape(625,1)
  z2 = np.tile(Z,(1,25)).reshape(625,1)

  #computing expectation
  exp_z1 = np.sum(z1* PI)
  exp_z2 = np.sum(z2* PI)


  return exp_z1, exp_z2


def q7():
  '''
  Loads the data and plots a color coded clustering of the conditional
  expectations.
  '''
  mat = loadmat('q7.mat')
  X, y = mat['x'], mat['y']
  

  conditions = np.array([conditional_prob_on_I(x) for x in X])
  plt.scatter(conditions[:,0], conditions[:,1], c=y.ravel(), cmap=plt.cm.jet)
  plt.show()



def load_model(model_file):
  '''
  Loads a default Bayesian network with latent variables (in this case, a
  variational autoencoder)
  '''

  with open('trained_mnist_model', 'rb') as infile:
    cpts = pkl.load(infile, encoding='bytes')

  model = {}
  model['prior_z1'] = cpts[0]
  model['prior_z2'] = cpts[1]
  model['cond_likelihood'] = cpts[2]

  return model


def main():

  global disc_z1, disc_z2
  n_disc_z = 25
  disc_z1 = np.linspace(-3, 3, n_disc_z)
  disc_z2 = np.linspace(-3, 3, n_disc_z)

  global bayes_net
  bayes_net = load_model('trained_mnist_model')

  # q4()
  # q5()
  # q6()
  q7()

  return

if __name__ == '__main__':

  main()
