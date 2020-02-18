# DeepPCSite
DeepPCSite is a deep neural network for predicting protein ligand sites using 3D point-cloud networks. Given a protein, the network develops features of the protein using chemical properties, feeds channels of these properties into an encoder, and uses the encoded representation to predict the ligandability of a certain region on a protein. The most ligandable region on the protein is considered the most likely to be a functional site of the protein.

The goals of DeepPCSite are to improve on the following:
- using a protein's spatial information directly without voxelizing the input
- finding lower dimensional representations for proteins and discovering which features are most important for a region to be a functional site
