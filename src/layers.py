import tensorflow.compat.v1 as tf

def weight_variables(shape, name):
	initial = tf.random.truncated_normal(shape = shape, mean = 0, stddev = 0.05)
	return tf.Variable(initial, name = name)

def get_chebyshev_coefficient(chebyshev_order, num_input_features, num_output_features):
	chebyshev_weights = dict()
	for i in range(chebyshev_order):
		initial = tf.random.truncated_normal(shape = (num_input_features, num_output_features), mean = 0, stddev = 0.05)
		chebyshev_weights['w' + str(i)] = tf.Variable(initial)
	return chebyshev_weights

def gcn_layer(input_PC, scaled_laplacian, num_points, num_input_features, num_output_features, chebyshev_order):
	bias_weight = weight_variables([num_output_features], name = 'bias_w')

	chebyshev_coefficient = get_chebyshev_coefficient(chebyshev_order, num_input_features, num_output_features)
	chebyshev_polynomial = []
	chebyshev_k_minus_1 = tf.matmul(scaled_laplacian, input_PC)
	chebyshev_k_minus_2 = input_PC
	chebyshev_polynomial.append(chebyshev_k_minus_2)
	chebyshev_polynomial.append(chebyshev_k_minus_1)

	# recursive formula for chebyshev polynomial approximation
	for i in range(2, chebyshev_order):
		chebyshev_k = 2 * tf.matmul(scaled_laplacian, chebyshev_k_minus_1) - chebyshev_k_minus_2
		chebyshev_polynomial.append(chebyshev_k)
		chebyshev_k_minus_2 = chebyshev_k_minus_1
		chebyshev_k_minus_1 = chebyshev_k

	chebyshev_output = []
	for i in range(chebyshev_order):
		weights = chebyshev_coefficient['w' + str(i)]
		chebyshev_polynomial_reshape = tf.reshape(chebyshev_polynomial[i], [-1, num_input_features])

		output = tf.matmul(chebyshev_polynomial_reshape, weights)
		output = tf.reshape(output, [-1, num_points, num_output_features])
		chebyshev_output.append(output)

	gcn_output = tf.add_n(chebyshev_output) + bias_weight
	gcn_output = tf.nn.relu(gcn_output)
	return gcn_output

def global_pooling(gcn_output, num_features):
	mean, var = tf.nn.moments(gcn_output, axes = [1])
	max_f = tf.reduce_max(gcn_output, axis = [1])
	pooling_output = tf.concat([max_f, var], axis = 1)
	return pooling_output

def fully_connected(features, num_input_features, num_output_features):
	weight_fc = weight_variables([num_input_features, num_output_features], name = 'weight_fc')
	bias_fc = weight_variables([num_output_features], name = 'bias_fc')
	output_fc = tf.matmul(features, weight_fc) + bias_fc
	return output_fc



