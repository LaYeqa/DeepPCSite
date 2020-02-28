class Hyperparameters():
	def __init__(self):
		# point-cloud features
		self.num_points = 300
		self.neighbor_number = 20

		# number of features per layer
		self.num_gcn_1_output_features = 1000
		self.num_gcn_2_output_features = 1000
		self.num_fc_1_output_features = 500
		
		# chebyshev polynomials
		self.chebyshev_1_order = 4
		self.chebyshev_2_order = 3

		# dropout
		self.keep_prob_1 = 0.9
		self.keep_prob_2 = 0.55

		# minibatches
		self.batch_size = 64

		# general
		self.max_epoch = 300
		self.learning_rate = 12e-4
		