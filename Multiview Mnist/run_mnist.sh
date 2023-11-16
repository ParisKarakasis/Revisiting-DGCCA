#!/bin/bash

# Download MNIST data
if [ ! -f "mnist_all.mat" ]
then
    echo "Downloading dataset ..."
    wget https://cs.nyu.edu/~roweis/data/mnist_all.mat --no-check-certificate
fi

# Create two-view Multiview MNIST
if [ ! -f "MNIST_dataset.mat" ]
then
    matlab -nodisplay -nosplash -nodesktop -r "run('create_MNIST_dataset.m'); exit;"
fi

# Run the algorithm
python crossencoder_MNIST.py 105 10
