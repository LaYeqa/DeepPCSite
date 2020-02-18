import pandas as pd 
import shutil
import util

TRAIN_SIZE = 12034
DEV_SIZE = 2000
TEST_SIZE = 2000
NUM_PROTEINS = 16034

def read_proteins():

	util.pretty_print("CONVERTING PROTEIN MOL2 FILES TO CSV FILES")

	seen_proteins = 0
	protein = []
	with open('scpdb/scPDB.mol2') as scpdb:
		for line in scpdb:
			if line == '@<TRIPOS>MOLECULE\n':
				if len(protein) != 0: # doesn't make file on first pass
					util.protein_to_csv(protein)
					seen_proteins = seen_proteins + 1
				protein.clear() # save storage by clearing current protein

				if seen_proteins % 500 == 0:
					print("Number of proteins seen: ", seen_proteins)
			protein.append(line)
		# handles the last protein to add into a csv file
		util.protein_to_csv(protein)
		seen_proteins = seen_proteins + 1
		protein.clear()

	util.pretty_print("TOTAL NUMBER OF PROTEINS SEEN: %s" %seen_proteins)

def split_files():

	util.pretty_print("SPLITTING ALL FILES INTO TRAIN / DEV / TEST SETS")

	# import scPDB ligands into a dataframe
	scpdb_ligands_df = pd.read_csv('scpdb/scpdb_ligands.tsv', sep = '\t')

	# get a list of all of the protein names and shuffle them
	protein_names = scpdb_ligands_df['PDB_ID'].tolist()
	protein_sites = scpdb_ligands_df['Site_Number'].tolist()

	# shuffle both lists in the same order
	protein_names, protein_sites = util.shuffle_two_lists(protein_names, protein_sites)

	# divide the list into train, dev, and test sets
	for i in range(len(protein_names)):
		filename = 'scpdb/all/%s_%s.csv' %(protein_names[i], protein_sites[i])
		if i < TRAIN_SIZE:
			shutil.move(filename, 'scpdb/train')
		elif i < (TRAIN_SIZE + DEV_SIZE):
			shutil.move(filename, 'scpdb/dev')
		else:
			shutil.move(filename, 'scpdb/test')

if __name__ == '__main__':
	read_proteins()
	split_files()
