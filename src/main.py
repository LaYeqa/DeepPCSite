import read_data
import model
import utils
from Hyperparameters import Hyperparameters

import tensorflow.compat.v1 as tf
import numpy as np



hyperparameters = Hyperparameters()

with tf.Graph().as_default():
	# =========================================================================================================
	# BUILD MODEL
	# =========================================================================================================
	train_operation = model.model_architecture(hyperparameters)

	# =========================================================================================================
	# LOAD DATA
	# =========================================================================================================
	input_train, train_label, input_test, test_label = read_data.load_data(hyperparameters.num_points)
	scaled_laplacian_train, scaled_laplacian_test = read_data.prepare_data(input_train, input_test, hyperparameters.num_neighhbors, hyperparameters.num_points)
	
	# =========================================================================================================
	# TRAIN MODEL
	# =========================================================================================================
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	saver = tf.train.Saver()

	learning_rate = hyperparameters.learning_rate

	save_model_path = '../model/'
	weight_dict = utils.weight_dict_fc(train_label, hyperparameters)
	test_label_whole = []
	for i in range(len(test_label)):
		labels = test_label[i]
		[test_label_whole.append(j) for j in labels]
	test_label_whole = np.asarray(test_label_whole)

	for epoch in range(hyperparameters.max_epoch):
		utils.pretty_print("EPOCH %i" %epoch)

		# learning rate decay
		if (epoch % 20 == 0):
			learning_rate /= 2
		learning_rate = np.max([learning_rate, 10e-6])

		train_avg_loss, train_avg_accuracy, loss_reg_avg = model.train_one_epoch(input_train, scaled_laplacian_train, train_label, hyperparameters, sess, train_operation, weight_dict, learning_rate)

		save = saver.save(sess, save_model_path)

		utils.pretty_print("AVERAGE LOSS: %i,\t AVERAGE L2 LOSS: %i,\t AVERAGE ACCURACY: %i,\t" %(train_avg_loss, train_avg_accuracy, loss_reg_avg))

		test_average_loss, test_average_accuracy, test_predict = model.evaluate_one_epoch(input_test, scaled_laplacian_test, test_label, hyperparameters, sess, train_operation)



























