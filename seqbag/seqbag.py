import pandas as pd
import numpy as np
import tensorflow as tf
import os

def fully_conn(x_tensor, num_outputs):
		_, num_inputs = x_tensor.get_shape().as_list()
		W = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=np.sqrt(2.0/num_inputs)))
		b = tf.Variable(tf.zeros([num_outputs]))
		return tf.add(tf.matmul(x_tensor, W), b)

def output_layer(x_tensor, num_outputs):
		_, num_inputs = x_tensor.get_shape().as_list()
		W = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=np.sqrt(1.0/num_inputs)))
		b = tf.Variable(tf.zeros([num_outputs]))
		return tf.add(tf.matmul(x_tensor, W), b)

def MLP(x, hidden_sizes, keep_prob):
		for i in hidden_sizes:
				x = fully_conn(x, i)
				x = tf.nn.relu(x)
				x = tf.nn.dropout(x, keep_prob=keep_prob)
		
		# last hidden -> output
		x = output_layer(x, 1)
		
		return x

def train_neural_net(session, optimizer,  keep_probibility, feature_batch, label_batch):
		session.run(optimizer, feed_dict={features: feature_batch,
																			labels: label_batch,
																			keep_prob: keep_probibility})

def batch_features_labels_w_prev(features, prev_pred, labels, batch_size):
		for start in range(0, len(features), batch_size):
				end = min(start + batch_size, len(features))
				yield features[start:end], prev_pred[start:end], labels[start:end]

def print_stats_w_prev(session, feature_batch, prev_batch, label_batch, cost, accuracy):
		loss = session.run(cost, feed_dict={features: feature_batch,
																				labels: label_batch,
																				prev_pred: prev_batch,
																				keep_prob: 1.0})
		acc = session.run(accuracy, feed_dict={features: feature_batch,
																					 labels: label_batch,
																					 prev_pred: prev_batch,
																					 keep_prob: 1.0})
		print('Loss: {:>10.6f} | Acc: {:>10.6f} '.format(loss, acc), end='')

def sequential_bagging_train(X_train, y_train, X_val, y_val, model_path, num_networks=3, hidden_sizes=[5, 5], keep_probability=0.7, Lambda=0.7, epochs=50, batch_size=256):
	tf.reset_default_graph()

	# create directory to store the ensemble
	ensemble_dir = './seq_bag_models/'
	if not os.path.exists('./seq_bag_models/'):
			os.makedirs('./seq_bag_models/')

	feature_count = len(X_train[0])
	output_count = 1

	# dictionary to store the networks created along the way
	network = {}

	# np.array, should be the average (logerr_1 + ... + logerr_i)/i
	prev_prediction = np.zeros_like(y_train.reshape(-1, 1))
	prev_prediction_val = np.zeros_like(y_val.reshape(-1, 1))

	for i in range(1, num_networks + 1):

			model_dir = 'model_{:1}/'.format(i)
			if not os.path.exists(ensemble_dir + model_dir):
					os.makedirs(ensemble_dir + model_dir)

			print('--------------------------------------------------')
			print('Training Network {:1} ...'.format(i))
			
			# inputs
			features = tf.placeholder(tf.float32, [None, feature_count], name='x_{:1}'.format(i))
			labels = tf.placeholder(tf.float32, [None, output_count], name='y_{:1}'.format(i))
			prev_pred = tf.placeholder(tf.float32, [None, output_count], name='prev_pred_{:1}'.format(i))
			keep_prob = tf.placeholder(tf.float32, name='keep_prob_{:1}'.format(i))
			
			# Model
			network[i] = MLP(features, hidden_sizes, keep_probability)
			
			# Name network_i output Tensor, so that is can be loaded from disk after training
			network[i] = tf.identity(network[i], name='network_{:1}'.format(i))
			
			# Loss and Optimizer
			cost = tf.losses.mean_squared_error(labels=labels, predictions=network[i])
			cost = tf.subtract(cost, tf.multiply(0.0 if (i == 1) else Lambda, tf.losses.mean_squared_error(labels=prev_pred, predictions=network[i])))
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999).minimize(cost)

			# Accuracy
			total_error = tf.reduce_sum(tf.square(tf.subtract(labels, tf.reduce_mean(labels))))
			unexplained_error = tf.reduce_sum(tf.square(tf.subtract(labels, ((i - 1)*prev_pred + network[i])/i)))
			accuracy = tf.subtract(1.0, tf.div(unexplained_error, total_error))
			
			# Sample with replacement (i > 1)
			if i == 1:
					X_train_, y_train_, prev_prediction_ = X_train, y_train, prev_prediction
			else:
					sample_w_rep = np.random.randint(0, len(X_train), len(X_train))
					X_train_ = X_train[sample_w_rep]
					y_train_ = y_train[sample_w_rep]
					prev_prediction_ = prev_prediction[sample_w_rep]
			
			# path to which we will save this network
			save_model_path = ensemble_dir + model_dir +  model_path # + '_{:1}'.format(i)

			with tf.Session() as sess:
					sess.run(tf.global_variables_initializer())

					# train network i
					for epoch in range(epochs):
							for batch_features, batch_prev_pred, batch_labels in batch_features_labels_w_prev(X_train_, prev_prediction_, y_train_.reshape(-1, 1), batch_size):
									sess.run(optimizer, feed_dict={features: batch_features,
																								 prev_pred: batch_prev_pred,
																								 labels: batch_labels,
																								 keep_prob: keep_probability})
							print('Epoch {:>4} ||  train  | '.format(epoch + 1), end='')
							# print_stats_w_prev(sess, X_train_, prev_prediction_, y_train_.reshape(-1, 1), cost, accuracy)
							loss = sess.run(cost, feed_dict={features: X_train_,
																									labels: y_train_.reshape(-1, 1),
																									prev_pred: prev_prediction_,
																									keep_prob: 1.0})
							acc = sess.run(accuracy, feed_dict={features: X_train_,
																										 labels: y_train_.reshape(-1, 1),
																										 prev_pred: prev_prediction_,
																										 keep_prob: 1.0})
							print('Loss: {:>10.6f} | Acc: {:>10.6f} '.format(loss, acc), end='')

							print('||  val  | '.format(epoch + 1), end='')
							# print_stats_w_prev(sess, X_val, prev_prediction_val, y_val.reshape(-1, 1), cost, accuracy)
							loss = sess.run(cost, feed_dict={features: X_val,
																									labels: y_val.reshape(-1, 1),
																									prev_pred: prev_prediction_val,
																									keep_prob: 1.0})
							acc = sess.run(accuracy, feed_dict={features: X_val,
																										 labels: y_val.reshape(-1, 1),
																										 prev_pred: prev_prediction_val,
																										 keep_prob: 1.0})
							print('Loss: {:>10.6f} | Acc: {:>10.6f} '.format(loss, acc), end='')
							print('')
					
					# update the aggregate logerror for the training set (i.e. prev_prediction)
					prev_prediction *= (i - 1)
					prev_prediction += sess.run(network[i], feed_dict={features: X_train, keep_prob: 1.0})
					prev_prediction /= i
					
					prev_prediction_val *= (i - 1)
					prev_prediction_val += sess.run(network[i], feed_dict={features: X_val, keep_prob: 1.0})
					prev_prediction_val /= i

					saver = tf.train.Saver()
					save_path = saver.save(sess, save_model_path)

def sequential_bagging_predict(X, model_path, num_networks=3):
	'''
	inputs:
			X:            numpy array of shape [num_data_points, num_features] 
										for the ensemble to predict on
			model_path:   a path to the the directory where the models in 
										the ensemble are stored
			num_networks: the number of networks in the ensemble

	returns:
			ensemble_logerror: ...
			network_outputs: ...
	'''

	ensemble_dir = './seq_bag_models/'
	network_outputs = {}
	for i in range(1, num_networks + 1):
			model_dir = 'model_{:1}/'.format(i)
			save_model_path = ensemble_dir + model_dir + model_path #+ '_{:1}'.format(i)

			loaded_graph = tf.Graph()

			with tf.Session(graph=loaded_graph) as sess:
					# load model
					loader = tf.train.import_meta_graph(save_model_path + '.meta')
					loader.restore(sess, save_model_path)
			
					# tensors from the loaded model
					loaded_x = loaded_graph.get_tensor_by_name('x_{:1}:0'.format(i))
					loaded_y = loaded_graph.get_tensor_by_name('y_{:1}:0'.format(i))
					loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob_{:1}:0'.format(i))
					loaded_logerror = loaded_graph.get_tensor_by_name('network_{:1}:0'.format(i))
			
					network_outputs[i] = sess.run(loaded_logerror, 
																				feed_dict={loaded_x: X, 
																									 loaded_keep_prob: 1.0})
					
	ensemble_logerror = np.zeros_like(network_outputs[1])
	for network in network_outputs:
			ensemble_logerror += network_outputs[network]/num_networks

	return ensemble_logerror, network_outputs
