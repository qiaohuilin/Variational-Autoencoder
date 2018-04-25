# Auto-Encoding Variational Bayes
This package is an implementation of *Auto-Encoding Variational Bayes*  by D. Kingma and Welling (2013) . This code used sigmoids and adagrad as the paper but provides more optimizers as SGD,SGD-momentum, ADAM, RMSprop. These choices provides similar results and convergence performance with similar speed.

Our version of code is based on Theano and Tensorflow to make the code much faster. 

**To run the MNIST experiment (train and test)**:

python AEVB-runExample1-Binary.py

**To run the FreyFaces experiment  (train and test)**:

python AEVB-runExample2-Continuous.py

MNIST Example takes around 5 min. FREYFACES around 25 min. 

**If one has his/her own dataset**, 
Please feed the data to AEVB-Source.py according to the instruction inside. 

**We also provide notebook version of code and result**

The code is MIT licensed.

© 2018 GitHub, Inc.