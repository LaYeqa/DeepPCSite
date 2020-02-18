import csv
import re
import random

# prints text with a topbar and bottom bar
def pretty_print(text):
	print("-----------------------------------------------")
	print(text)
	print("-----------------------------------------------\n")

# removes all occurrences of the elements in chars from list
def remove_all(lst, chars):
	for char in chars:
		while char in lst:
			lst.remove(char)
	return lst

# shuffles two lists in the same random order
def shuffle_two_lists(list1, list2):
	zipped = list(zip(list1, list2))
	random.shuffle(zipped)
	return zip(*zipped)

# find the site number of a given protein
# use this to differentiate same protein different binding sites
def get_site_number(line):
    # find the protein part
    protein_index = line.lower().find('_protein')

    # if it's a number with two digits
    if line[protein_index - 2].isdigit():
    	return line[protein_index - 2 : protein_index]
    else: # just one digit
    	if not line[protein_index - 1].isdigit():
    		print(line)
    	return line[protein_index - 1]
    

# converts protein text into a csv file
def protein_to_csv(protein):
	# get the protein name
	protein_name = protein[1][:4].lower()
	site_number = get_site_number(protein[1])
	assert(site_number.isdigit())
	# find the atoms portion of the protein
	atoms_index = protein.index('@<TRIPOS>ATOM\n') + 1
	bonds_index = protein.index('@<TRIPOS>BOND\n')

	# create csv file with protein atom information
	with open('scpdb/all/%s_%s.csv' %(protein_name, site_number), 'w') as protein_csv:
		writer = csv.writer(protein_csv, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)

		# header for all entries
		writer.writerow(['atom_id', 'atom_name', 'x_coord', 'y_coord', 'z_coord', 'atom_type', 'subst_id', 'subst_name', 'charge'])

		# add each atom's information to the csvfile
		for atom in protein[atoms_index : bonds_index]:
			writer.writerow(remove_all(re.split(' |\t', atom.strip()), ['', ' ', '\t']))