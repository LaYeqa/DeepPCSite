# DeepPCSite
DeepPCSite is a deep neural network for predicting protein ligand sites using 3D point-cloud networks. Given a protein, the network develops features of the protein using chemical properties, feeds channels of these properties into an encoder, and uses the encoded representation to predict the ligandability of a certain region on a protein. The most ligandable region on the protein is considered the most likely to be a functional site of the protein.

The goals of DeepPCSite are to improve on the following:
- using a protein's spatial information directly without voxelizing the input to reduce data sparsity
- finding lower dimensional representations for proteins and discovering which features are most important for a region to be a functional site

# How It Works
## Proteins as Point-Clouds
One of the most common ways of representing proteins to determine functional sites is through voxelization. This approach allows one to use filled and empty voxels to determine the spatial relationship between atoms in the protein region. However, this data structure is very sparse and requires a significant amount of preprocessing. DeepPCSite posits another approach by embedding protein regions as their most unprocessed state: 3D point-clouds. Each atom represents a point in the cloud, and spatial features are already intrinsic to the data structure. 

## Deep Learning on Point-Clouds
In order to perform deep learning on point-clouds, DeepPCSite uses a graph convolutional neural network to learn spatial and spectral features of the protein site. Each atom, thought of as a node in a graph, is given an edge with its k-nearest neighbors, and the Laplacian of the graph is used for graph learning. The coordinates in the point-cloud, along with other chemoinformatical properties (atom type, aromaticity, charge, hydrogen bond donor/acceptor, etc.) are thought of as signals on the graph for the convolution step. This way, each of the features that would be learned through preprocessing a voxelized approached are localized to each atom.
