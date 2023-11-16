#!/bin/bash

# Download XRMB data
if [ ! -f "XRMBf2KALDI_window7_single1.mat" ]
then
    echo "Downloading dataset ..."
    wget http://ttic.edu/livescu/XRMB_data/full/XRMBf2KALDI_window7_single1.mat --no-check-certificate
fi
if [ ! -f "XRMBf2KALDI_window7_single2.mat" ]
then
    echo "Downloading dataset ..."
    wget http://ttic.edu/livescu/XRMB_data/full/XRMBf2KALDI_window7_single2.mat --no-check-certificate
fi

# Create reduced XRMB datset
if [ ! -f "XRMB_reduced.mat" ]
then
    matlab -nodisplay -nosplash -nodesktop -r "run('Get_reduced_dataset_and_labels.m'); exit;"
fi

# Run the algorithm
python crossencoder_XRMB.py 105 30
