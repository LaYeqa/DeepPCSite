import numpy as np
import pandas as pd
import csv
import utils

NUM_POINTS = 300
BOX_WIDTH = 20.0
BOX_GAP = 5.0

def get_geometric_center(atoms):
	"""Gets the geometric center of a protein binding site."""

	# gets the site extrema and finds the midpoint
	box_min, box_max = get_protein_extrema(atoms)
	return (box_max + box_min) / 2


def get_label(box_center, site_center):
	"""Returns a boolean (1 / 0) detailing whether the current bounding box is within 5 angstroms of the site center."""

	return int(np.linalg.norm(site_center - box_center) <= 5.0)


def get_box_from_center(center):
	box_min = center - BOX_WIDTH / 2
	box_max = center + BOX_WIDTH / 2
	return box_min, box_max


def is_valid_box(box_center, protein):
	"""Returns whether or not the box has enough atoms to be passed into the net."""

	box_min, box_max = get_box_from_center(box_center)
	atoms_in_box = np.logical_and(protein - box_min >= 0, box_max - protein >= 0)
	num_atoms = np.count_nonzero(np.count_nonzero(atoms_in_box, axis = 1) == 3)
	return num_atoms >= NUM_POINTS


def get_protein_extrema(atoms):
	"""Gets the minimum and the maximum coordinates in the atom space."""

	box_min = np.amin(atoms, axis = 0)
	box_max = np.amax(atoms, axis = 0)
	return box_min, box_max


def get_bounding_boxes(protein):
	"""Returns a list of the potential bounding boxes based on the protein."""

	protein_min, protein_max = get_protein_extrema(protein)
	min_center = protein_min + BOX_WIDTH / 2
	max_center = protein_max - BOX_WIDTH / 2

	boxes = []
	for x in np.arange(min_center[0], max_center[0], BOX_GAP):
		for y in np.arange(min_center[1], max_center[1], BOX_GAP):
			for z in np.arange(min_center[2], max_center[2], BOX_GAP):
				boxes.append([x, y, z])
	return boxes


if __name__ == '__main__':

	# create the file to hold the labels
	with open('../data/labels.csv', 'w') as labels:
		writer = csv.writer(labels, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

		# header for all rows
		writer.writerow(['pdb_id', 'site_number', 'box_center_x', 'box_center_y', 'box_center_z', 'site_center_x', 'site_center_y', 'site_center_z', 'label'])

		# iterate over all proteins in the csv file
		sites = pd.read_csv('../data/site/scpdb_sites.tsv', sep = '\t', error_bad_lines = False)
		for index, row in sites.iterrows():
			if index % 500 == 0:
				utils.pretty_print("Number of proteins seen: %d" %index)

			protein_name, site_number = row['PDB_ID'], row['Site_Number']

			site_file = '../data/site/csv/%s_%s.csv' %(protein_name, site_number)
			site = pd.read_csv(site_file, error_bad_lines = False)
			site_center = get_geometric_center(site[['x_coord', 'y_coord', 'z_coord']].to_numpy())

			protein_file = '../data/protein/csv/%s_%s.csv' %(protein_name, site_number)
			protein = pd.read_csv(protein_file, error_bad_lines = False)[['x_coord', 'y_coord', 'z_coord']].to_numpy()
			
			boxes = get_bounding_boxes(protein) # have get_bounding_boxes use get_protein_extrema

			for box_center in boxes:
				box_center = np.asarray(box_center)
				if is_valid_box(box_center, protein):
					label = get_label(box_center, site_center)
					writer.writerow([protein_name, site_number, box_center[0], box_center[1], box_center[2], site_center[0], site_center[1], site_center[2], label])


