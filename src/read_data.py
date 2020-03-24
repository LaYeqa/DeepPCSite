import utils

import numpy as np
import pandas as pd



BOX_WIDTH = 20.0
def get_box_from_center(center):
	box_min = center - BOX_WIDTH / 2
	box_max = center + BOX_WIDTH / 2
	return box_min, box_max


def uniform_sampling(atoms, num_points):
	mask = np.random.choice(len(atoms), num_points, replace = False)
	return atoms[mask]


def get_features(atoms):
	num_features = 5 # x, y, z, atom, charge
	atom_enum = {'H': 1, 'C': 2, 'O': 3, 'N': 4, 'P': 5, 'S': 6}
	features = np.zeros((len(atoms), num_features))

	atom_col = np.zeros((len(atoms), 1))
	for i in range(len(atoms)):
		# charge index = 5
		atom_name = atoms[i][5][:atoms[i][5].find('.')] if atoms[i][5].find('.') != -1 else atoms[i][5]
		if atom_name in atom_enum:
			atom_col[i] = atom_enum[atom_name]
	charge_col = atoms[:, -1].reshape((len(atoms), -1))

	features = np.concatenate((atoms[:, 2:5], atom_col, charge_col), axis = 1)
	return features


def get_atoms_from_center(protein_name, center, num_points):
	box_min, box_max = get_box_from_center(center)
	protein = pd.read_csv('../data/protein/csv/%s.csv' %protein_name, error_bad_lines = False)
	protein_atoms = protein[['x_coord', 'y_coord', 'z_coord']].to_numpy()

	coords_in_box = np.logical_and(protein_atoms - box_min >= 0, box_max - protein_atoms >= 0)
	atoms_in_box = np.count_nonzero(coords_in_box, axis = 1) == 3
	atoms_in_box_indices = np.where(atoms_in_box)[0]
	assert(len(atoms_in_box_indices) >= num_points)

	atoms = protein.iloc[atoms_in_box_indices].to_numpy()
	atoms_samples = uniform_sampling(atoms, num_points)
	feature_samples = get_features(atoms_samples)

	return feature_samples


def get_atoms(proteins, num_points):
	pc = [] # m x num_points x num_features dimensional
	labels = [] # 1 x m dimensional
	total_labels = pd.read_csv('../data/labels.csv', error_bad_lines = False)

	print("total positive: ", np.sum(total_labels['label'].to_numpy()))

	idx = 0
	for protein in proteins:
		if idx % 100 == 0:
			print("\tREADING %i OF %i PROTEINS..." %(idx, len(proteins)))
		idx += 1

		protein = protein.strip()
		protein_id = protein[:protein.find('_')]
		protein_site = int(protein[protein.find('_') + 1:])
		protein_labels = total_labels[(total_labels['pdb_id'] == protein_id) & (total_labels['site_number'] == protein_site)]
		centers = protein_labels[['box_center_x', 'box_center_y', 'box_center_z', 'label']].to_numpy()

		for center in centers:
			atoms = get_atoms_from_center(protein, center[0:3], num_points)
			pc.append(atoms) 
			labels.append(center[-1]) 

	return np.array(pc), np.array(labels)


NUM_TRAIN_PROTEINS = 12000
NUM_TEST_PROTEINS = 4034
def load_data(num_points):

	# load the train and test protein names
	with open('../data/train.txt', 'r') as train_file:
		train_proteins = train_file.readlines() # note: includes newlines
	with open('../data/test.txt', 'r') as test_file:
		test_proteins = test_file.readlines()

	utils.pretty_print("LOADING THE TRAINING SET...")
	input_train, train_labels = get_atoms(train_proteins, num_points)
	utils.pretty_print("LOADING THE TESTING SET...")
	input_test, test_labels = get_atoms(test_proteins, num_points)

	utils.save_numpy([input_train, train_labels, input_test, test_labels], ['input_train', 'train_labels', 'input_test', 'test_labels'])
	return input_train, train_labels, input_test, test_labels

import scipy
def get_adjacency(distances, indices):
	"""Returns the adjacency matrix of a k-nearest neighbors graph."""
	m, k = distances.shape
	sigma2 = np.mean(distances[:, -1]) ** 2
	distances = np.exp(-distances ** 2 / sigma2)

	I = np.arange(0, m).repeat(k)
	J = indices.reshape(m * k)
	V = distances.reshape(m * k)
	W = scipy.sparse.coo_matrix((V, (I, J)), shape = (m, m))
	W.setdiag(0)

	bigger = W.T > W
	W = W - W.multiply(bigger) + W.T.multiply(bigger)
	return W


def normalize_adj(adjacency):
    adacency = scipy.sparse.coo_matrix(adjacency)
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_scaled_laplacian(adjacency):
    adj_normalized = normalize_adj(adjacency)
    laplacian = scipy.sparse.eye(adjacency.shape[0]) - adj_normalized
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adjacency.shape[0])
    return scaled_laplacian


from scipy.spatial import cKDTree
def prepare_graph(input_data, num_neighbors, num_points):
	scaled_laplacians = [] # m x NP x NP dimensional
	for i in range(len(input_data)):
		if i % 1000 == 0:
			utils.pretty_print("READING %i OF %i GRAPHS" %(i, len(input_data)))

		coords = input_data[i][:3]
		ckdtree = cKDTree(coords)
		distances, indices = tree.query(coords, k = num_neighbors)
		adjacency = get_adjacency(distances, indices)
		scaled_laplacian = get_scaled_laplacian(adjacency)
		flatten_laplacian = scaled_laplacian.reshape((1, num_points * num_points))
		scaled_laplacians.append(flatten_laplacian)
	return np.array(scaled_laplacians)



def prepare_data(input_train, input_test, num_neighbors, num_points):
	utils.pretty_print("PREPARING THE DATA...")
	scaled_laplacian_train = prepare_graph(input_train, num_neighbors, num_points)
	scaled_laplacian_test = prepare_graph(input_test, num_neighbors, num_points)
	utils.save_numpy([scaled_laplacian_train, scaled_laplacian_test], ['scaled_laplacian_train', 'scaled_laplacian_test'])
	return scaled_laplacian_train, scaled_laplacian_test



