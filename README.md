# DeepPCSite
DeepPCSite is a deep neural network for predicting protein ligand sites using 3D point-cloud networks. Given a protein, the network develops features of the protein using chemical properties, feeds channels of these properties into an encoder, and uses the encoded representation to predict the ligandability of a certain region on a protein. The most ligandable region on the protein is considered the most likely to be a functional site of the protein.

The goals of DeepPCSite are to improve on the following:
- using a protein's spatial information directly without voxelizing the input to reduce data sparsity
- finding lower dimensional representations for proteins and discovering which features are most important for a region to be a functional site

# HOW IT WORKS
## Proteins as Point Clouds
DeepPCSite embeds protein data in their most unprocessed state: 3D point-clouds. Each atom represents a point in the cloud. These point-clouds are stored as graph adjacency matrices, where each atom is a vertex whose k-nearest neighbors in space constitute its neighborhood. Each vertex contains a series of features, which can be thought of as signals along the graph. 
## Protein Autoencoders
The deep network first reduces the dimensionality of the input through a variational graph autoencoder (VGAE).
