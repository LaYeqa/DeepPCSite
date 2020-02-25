import pandas as pd 
import shutil
import utils
import csv
import re

def get_site_number(line, filetype):
	"""Finds the site number of a given protein in the mol2 file."""
	
    # find the protein part
	protein_index = line.lower().find('_%s' %filetype)

	# if it's a number with two digits
	if line[protein_index - 2].isdigit():
		return line[protein_index - 2 : protein_index]
	else: # just one digit
		return line[protein_index - 1]


def mol2_atoms_to_csv(mol2, filetype):
	"""Converts a protein site into a csv file."""

	# get the protein name
	mol2_name = mol2[1][:4].lower()
	site_number = get_site_number(mol2[1], filetype)
	assert(site_number.isdigit())

	# find the atoms portion of the protein
	atoms_index = mol2.index('@<TRIPOS>ATOM\n') + 1
	bonds_index = mol2.index('@<TRIPOS>BOND\n')

	# create csv file with mol2 atom information
	with open('../data/%s/csv/%s_%s.csv' %(filetype, mol2_name, site_number), 'w') as mol2_csv:
		writer = csv.writer(mol2_csv, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

		# header for all entries
		writer.writerow(['atom_id', 'atom_name', 'x_coord', 'y_coord', 'z_coord', 'atom_type', 'subst_id', 'subst_name', 'charge'])

		# add each atom's information to the csvfile
		for atom in mol2[atoms_index : bonds_index]:
			writer.writerow(utils.remove_all(re.split(' |\t', atom.strip()), ['', ' ', '\t']))


def read_proteins(filetype):
	"""Reads the proteins in the proteins.mol2 file and converts them to csv files."""	

	seen_proteins = 0
	protein = []
	with open('../data/%s/scpdb.mol2' %filetype) as scpdb:
		for line in scpdb:
			if line == '@<TRIPOS>MOLECULE\n':
				if len(protein) != 0: # doesn't make file on first pass
					mol2_atoms_to_csv(protein, filetype)
					seen_proteins = seen_proteins + 1
				protein.clear() # save storage by clearing current protein

				if seen_proteins % 500 == 0:
					print("Number of proteins seen: ", seen_proteins)
			protein.append(line)
		# handles the last protein to add into a csv file
		mol2_atoms_to_csv(protein, filetype)
		seen_proteins = seen_proteins + 1
		protein.clear()

	utils.pretty_print("TOTAL NUMBER OF PROTEINS SEEN: %s" %seen_proteins)


def split_files():
	"""Splits the mol2 csv files into train, test, and dev sets.""" 

	# import scPDB ligand binding sites into a dataframe
	scpdb_ligands_df = pd.read_csv('../data/site/sites.tsv', sep = '\t')

	# get a list of all of the protein names and shuffle them
	protein_names = scpdb_ligands_df['PDB_ID'].tolist()
	protein_sites = scpdb_ligands_df['Site_Number'].tolist()

	# shuffle both lists in the same order
	protein_names, protein_sites = util.shuffle_two_lists(protein_names, protein_sites)

	# divide the list into train, dev, and test sets
	for i in range(len(protein_names)):
		filename = '../data/protein/csv/%s_%s.csv' %(protein_names[i], protein_sites[i])
		if i < TRAIN_SIZE:
			shutil.move(filename, '../data/protein/train')
		elif i < (TRAIN_SIZE + DEV_SIZE):
			shutil.move(filename, '../data/protein/dev')
		else:
			shutil.move(filename, '../data/protein/test')

if __name__ == '__main__':
	utils.pretty_print("CONVERTING MOL2 FILES TO CSV FILES")
	read_proteins('site')

	util.pretty_print("SPLITTING ALL FILES INTO TRAIN / DEV / TEST SETS")
	split_files()
