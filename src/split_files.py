import pandas as pd
import numpy as np

if __name__ == '__main__':
	proteins = pd.read_csv('../data/site/scpdb_sites.tsv', sep = '\t', error_bad_lines = False)['PDB_ID'].to_numpy()
	np.random.shuffle(proteins)
	assert(len(proteins) == 16034)

	for i in range(len(proteins)):
		filetype = 'train' if i < 12000 else 'test'
		with open('../data/%s.txt' %filetype, 'a') as file:
			file.write(proteins[i] + '\n')