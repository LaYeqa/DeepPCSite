import utils
import layers

import tensorflow.compat.v1 as tf
import numpy as np

def model_architecture(hyperparameters):
	"""Sets the hyperparameters for the model."""

	input_pc = tf.placeholder(tf.float32,  [None, hyperparameters.num_points, hyperparameters.num_features])
	input_graph = tf.placeholder(tf.float32, [None, hyperparameters.num_points * hyperparameters.num_points])
	output_label = tf.placeholder(tf.float32)

	scaled_laplacian = tf.reshape(input_graph, [-1, hyperparameters.num_points, hyperparameters.num_points])

	weights = tf.placeholder(tf.float32, [None])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob_1 = tf.placeholder(tf.float32)
	keep_prob_2 = tf.placeholder(tf.float32)

	# first layer: graph convolution
	gcn_1 = layers.gcn_layer(input_pc, scaled_laplacian, hyperparameters.num_points, hyperparameters.num_features, hyperparameters.num_gcn_1_output_features, hyperparameters.chebyshev_1_order)
	gcn_1_output = tf.nn.dropout(gcn_1, rate = 1 - keep_prob_1)
	gcn_1_pool = layers.global_pooling(gcn_1_output, hyperparameters.num_gcn_1_output_features)

	# second layer: graph convolution on the output of gcn_1 before pooling
	gcn_2 = layers.gcn_layer(gcn_1_output, scaled_laplacian, hyperparameters.num_points, hyperparameters.num_gcn_1_output_features, hyperparameters.num_gcn_2_output_features, hyperparameters.chebyshev_2_order)
	gcn_2_output = tf.nn.dropout(gcn_2, rate = 1 - keep_prob_1)
	gcn_2_pool = layers.global_pooling(gcn_2_output, hyperparameters.num_gcn_2_output_features)

	# concatenate global features between gcn_1 and gcn_2
	global_features = tf.concat([gcn_1_pool, gcn_2_pool], axis = 1)
	global_features = tf.nn.dropout(global_features, rate = 1 - keep_prob_2)
	num_global_features = 2 * (hyperparameters.num_gcn_1_output_features + hyperparameters.num_gcn_2_output_features)

	# first fully connected layer at the end
	fc_1 = layers.fully_connected(global_features, num_global_features, hyperparameters.num_fc_1_output_features)
	fc_1 = tf.nn.relu(fc_1)
	fc_1 = tf.nn.dropout(fc_1, rate = 1 - keep_prob_2)

	# second fully connected layer
	fc_2 = layers.fully_connected(fc_1, hyperparameters.num_fc_1_output_features, hyperparameters.num_fc_2_output_features)


	# =========================================================================================================
	# LOSS AND BACKPROPAGATION
	# =========================================================================================================

	# loss
	predict_label = tf.nn.sigmoid(fc_2) >= 0.5
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = fc_2, labels = output_label)
	loss = tf.reduce_mean(tf.multiply(loss, weights))

	train_vars = tf.trainable_variables()
	loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'bias' not in v.name]) * 8e-6
	loss_total = loss + loss_reg

	correct_prediction = tf.equal(predict_label, (output_label == 1))
	accuracy = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(accuracy)

	train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_total)

	train_operation = {'train': train,
				  	   'loss': loss,
				  	   'loss_reg': loss_reg,
				  	   'loss_total': loss_total,
				  	   'accuracy': accuracy,
				  	   'input_pc': input_pc,
				  	   'input_graph': input_graph,
				  	   'output_label': output_label,
				  	   'weights': weights,
				  	   'predict_label': predict_label,
				  	   'keep_prob_1': keep_prob_1,
				  	   'keep_prob_2': keep_prob_2,
				  	   'learning_rate': learning_rate}

	return train_operation


def train_one_epoch(input_pc, input_graph, input_label, hyperparameters, sess, train_operation, weight_dict, learning_rate):
	batch_loss = []
	batch_accuracy = []
	batch_reg_loss = []
	minibatch_size = hyperparameters.minibatch_size
	for minibatch_id in range(len(input_pc) / minibatch_size):
		start = minibatch_id * minibatch_size
		end = start + minibatch_size

		minibatch_coords, minibatch_graph, minibatch_label = input_pc[start:end], input_graph[start:end], input_label[start:label]
		minibatch_weight = utils.uniform_weight(minibatch_label)
		minibatch_graph = minibatch_graph.todense()

		feed_dict = {train_operation['input_pc']: minibatch_coords,
					 train_operation['input_graph']: minibatch_graph,
					 train_operation['output_label']: minibatch_label,
					 train_operation['learning_rate']: learning_rate,
					 train_operation['weights']: minibatch_weight,
					 train_operation['keep_prob_1']: hyperparameters.keep_prob_1,
					 train_operation['keep_prob_2']: hyperparameters.keep_prob_2}

		train_opt, train_loss, train_accuracy, train_reg_loss = sess.run([train_operation['train'], train_operation['loss'], train_operation['accuracy'], train_operation['loss_reg']], feed_dict = feed_dict)
		
		batch_loss.append(train_loss)
		batch_accuracy.append(train_accuracy)
		batch_reg_loss.append(train_reg_loss)

	train_average_loss = np.mean(batch_loss)
	train_average_accuracy = np.mean(batch_accuracy)
	train_average_reg_loss = np.mean(batch_reg_loss)
	return train_average_loss, train_average_accuracy, train_average_reg_loss


def evaluate_one_epoch(input_pc, input_graph, input_label, hyperparameters, sess, train_operation):
	test_loss = []
	test_accuracy = []
	test_predictions = []
	minibatch_size = hyperparameters.minibatch_size
	for minibatch_id in range(len(input_pc) / minibatch_size):
		start = minibatch_id * minibatch_size
		end = start + minibatch_size

		minibatch_coords, minibatch_graph, minibatch_label = input_pc[start:end], input_graph[start:end], input_label[start:label]
		minibatch_weight = utils.uniform_weight(minibatch_label)
		minibatch_graph = minibatch_graph.todense()

		feed_dict = {train_operation['input_pc']: minibatch_coords,
					 train_operation['input_graph']: minibatch_graph,
					 train_operation['output_label']: minibatch_label,
					 train_operation['weights']: minibatch_weight,
					 train_operation['keep_prob_1']: 1.0,
					 train_operation['keep_prob_2']: 1.0}

		predictions, loss, accuracy = sess.run([train_operation['predict_label'], train_operation['loss'], train_operation['accuracy']], feed_dict = feed_dict)

		test_loss.append(loss)
		test_accuracy.append(accuracy)
		test_predictions.append(predictions)

	test_average_loss = np.mean(test_loss)
	test_average_accuracy = np.mean(test_accuracy)
	return test_average_loss, test_average_accuracy, test_predictions




