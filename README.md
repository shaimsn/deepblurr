# deepblurr

This repo contains code to implement our non-blind enhanced deconvolution scheme that uses a neural network to non-linearly combine 15 simply-reconstructed images into a sharper representation.

dataset_generation contains Matlab files to generate the dataset. The output dataset is stored in dataset_generation/stack_drinks_dataset/. Please read dataset_generation/README_dataset_gen.txt for more info on how to generate the dataset. 

nn_architecture contains code to train and evaluate the neural networks. The outputs from generate_dataset/stack_drinks_dataset/ are split into test, val and train sets, and put inside nn_architecture/data/wiener_stacks/. Please see nn_architecture/README_nn_arch.txt for more info on how to train the model and evaluate it.