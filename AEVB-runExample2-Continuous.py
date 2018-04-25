
# AEVB Algorithm  -Example2 -Continuous Case (FreyFaces)

import matplotlib.pyplot as plt
import numpy as np


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
warnings.simplefilter(action='ignore', category=ImportWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)


import theano
import theano.tensor as T


# Define Gradient Descent Functions

def sgd(cost, params, lr=0.001, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append((p, p - lr * g))
    return updates


def sgd_momentum(cost, params, lr=0.001, beta=0.9,epsilon=1e-6):  #no bias correction
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v= theano.shared(p.get_value() * 0.)
        vc= beta*v + (1-beta)*g
        updates.append((v,vc))
        updates.append((p, p - lr * vc))
    return updates


def adagrad(cost, params, lr=0.001, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = acc + g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def adam(cost, params, lr=0.001, epsilon=1e-6):  #no bias correction, T.sqrt if **0.5 not work
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        m=theano.shared(p.get_value()* 0.)
        v=theano.shared(p.get_value()* 0.)
        mc = beta1*m + (1-beta1)*grad(x)
        vc = beta2*v + (1-beta2)*grad(x)**2
        updates.append((m,mc))
        updates.append((v,vc))
        updates.append((p, p - lr * mc/(epsilon+vc**0.5)))
    return updates


def RMSprop(cost, params, lr=0.001, beta=0.9,epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v= theano.shared(p.get_value() * 0.)
        vc = beta*v + (1-beta)*g**2
        updates.append((v,vc))
        updates.append((p, p - lr * g/(epsilon+vc**(0.5))))
    return updates


'''gaussian decoder and freyface starts next:'''

# Continuous Dataset
# Read in dataset
from scipy.io import loadmat
ff = loadmat('frey_rawface.mat', squeeze_me=True, struct_as_record=False)
ff1 = ff["ff"].T

f_train=ff1[:1500,]
f_test=ff1[1501:,]


# Intializing Parameters


n_hidden = 100 # the size of hidden layers in MLP
n_latent = 2 # the dimension of z
n_input = f_train.shape[1] # the dimension of f's feature space
batch_size = 100
n_epochs = 100000

def init_w(shape):
    x = np.random.randn(*shape)
    float_x = np.asarray(x * 0.01, dtype=theano.config.floatX)
    return theano.shared(float_x)


# Gaussian Encoder -Parameters
# Gaussian MLP weights and biases (encoder)
# initialize \phi 
b3 = init_w((n_hidden, ))
b2 = init_w((n_latent, ))
b1 = init_w((n_latent, ))


W3 = init_w((n_input, n_hidden))
W2 = init_w((n_hidden, n_latent))
W1 = init_w((n_hidden, n_latent))


#Gaussian encoder
x = T.matrix("x")
h_encoder = T.tanh(T.dot(x, W3) + b3)
mu = T.dot(h_encoder, W1) + b1
log_sig2 = T.dot(h_encoder, W2) + b2
# This expression is simple (not an expectation) because we're using normal priors and posteriors
DKL = (1.0 + log_sig2 - mu**2 - T.exp(log_sig2)).sum(axis = 1)/2.0


# Parameters
# Gaussian MLP weights and biases (decoder)
# initialize \theta

b6 = init_w((n_hidden, ))
b5 = init_w((n_input, ))
b4 = init_w((n_input, ))


W6 = init_w((n_latent, n_hidden))
W5 = init_w((n_hidden, n_input))
W4 = init_w((n_hidden, n_input))



# Gaussian Decoder - Parameters
# Gaussian MLP weights and biases (decoder)
std_normal = T.matrix("std_normal") 
z = mu + T.sqrt(T.exp(log_sig2))*std_normal
h_decoder = T.tanh(T.dot(z, W6) + b6)
mu_prime = T.dot(h_decoder, W4) + b4
log_sig2_prime = T.dot(h_decoder, W5) + b5

log_likelihood_gaus= (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sig2_prime) - 0.5 * ((x - mu_prime)**2 / T.exp(log_sig2_prime))).sum(axis=1).mean(axis=0)



# Only the weight matrices W will be regularized (weight decay)
W = [W3, W1, W2, W6, W4, W5]
b = [b3, b1, b2, b6, b5, b4]
params = W + b


# Variational Lower Bound and cost  - objective function
lower_bound = (DKL + log_likelihood_gaus).mean()
cost = -lower_bound


# Define updates: which gradient descent to use
updates = adagrad(cost, params, lr=0.02)


# Compile the model
train_model = theano.function(inputs=[x, std_normal], 
                              outputs=cost, 
                              updates=updates,
                              mode='FAST_RUN',
                              allow_input_downcast=True)



# Train the model
np.random.seed(1)
training = []
for i in range(n_epochs):
    minibatch_train = [ f_train[j] for j in np.random.randint(0,f_train.shape[0],batch_size) ]

    train_cost = train_model(minibatch_train, np.random.normal(size = (batch_size, n_latent)))
    
    training.append(train_cost)
 


# Plot the lower bound
lb=[-i for i in training]
plt.ylabel("-Lower bound")
plt.xlabel("Minibatch (" + str(batch_size) + " samples)")
plt.plot(lb[500:],)


# Run,save and plot the likelihood

updates = adagrad(cost, params, lr=0.02)

train_model_likelihood = theano.function(inputs=[x, std_normal], 
                              outputs=log_likelihood_gaus, 
                              updates=updates,
                              mode='FAST_RUN',
                              allow_input_downcast=True)

training_likelihood = []

for i in range(n_epochs):
    minibatch_train = [ f_train[j] for j in np.random.randint(0,f_train.shape[0],batch_size) ]

    likelihood = train_model_likelihood(minibatch_train, np.random.normal(size = (batch_size, n_latent)))
    
    training_likelihood.append(likelihood)


# Plot the likelihood traceplot
plt.ylabel("likelihood")
plt.xlabel("Minibatch (" + str(batch_size) + " samples)")
plt.plot(training_likelihood[500:])


# Extract estimated parameters (W, b)
est_W3 = updates[1][0].get_value()
est_W1 = updates[3][0].get_value()
est_W2 = updates[5][0].get_value()
est_W6 = updates[7][0].get_value()
est_W4 = updates[9][0].get_value()
est_W5 = updates[11][0].get_value()

est_b3 = updates[13][0].get_value()
est_b1 = updates[15][0].get_value()
est_b2 = updates[17][0].get_value()
est_b6 = updates[19][0].get_value()
est_b4 = updates[21][0].get_value()
est_b5 = updates[23][0].get_value()


# Reconstruction on Test Set

# Gaussian Encoder on f_test
est_h_encoder = np.tanh(np.dot(f_test, est_W3) + est_b3)
phi_mu = np.dot(est_h_encoder, est_W1) + est_b1
phi_log_sig2 = np.dot(est_h_encoder, est_W2) + est_b2

z_test = phi_mu + np.sqrt(np.exp(phi_log_sig2))*np.random.randn(phi_log_sig2.shape[0], phi_log_sig2.shape[1])


# Gaussian Decoder on x_test
est_h_decoder=np.tanh(np.dot(z_test,est_W6)+ est_b6)
theta_mu = np.dot(est_h_decoder,est_W4) + est_b4
theta_log_sig2=np.dot(est_h_decoder, est_W5) + est_b5


# Prediction
f_new=np.random.normal(theta_mu,theta_log_sig2,[theta_mu.shape[0],theta_mu.shape[1]])


# Visualization
# View Original Test Set

f_test1 = f_test.reshape([464, 28, 20])
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(f_test1[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()


# View Reconstruction Result
f_new1 = f_new.reshape([464, 28, 20])
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(f_new1[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

