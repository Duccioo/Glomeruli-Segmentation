from datetime import datetime
import os
import sys
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from network import *
from utils import ImageReader, decode_labels, inv_preprocess, prepare_label, write_log

import notifica_telegram

"""
Training:
'poly' learning rate
different learning rates for different layers
"""



IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

class Model(object):

	def __init__(self, sess, conf):
		self.sess = sess
		self.conf = conf

	# train
	def train(self):
		self.train_setup()

		self.sess.run(tf.global_variables_initializer())

		last_checkpoint=0

		# Load the pre-trained model if provided
		if self.conf.pretrain_file is not None:
			self.load(self.loader, self.conf.pretrain_file)
			last_checkpoint=self.conf.last_checkpoint
		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		# log_var for tensorboard
		log_var = tf.Variable(0.0)
		summary_loss = [tf.summary.scalar("loss", log_var)]
		write_op = tf.summary.merge(summary_loss)
		#write_op = tf.summary.merge_all()
		#self.sess.run(tf.global_variables_initializer())


		# Train!
		for (step) in range(self.conf.num_steps+1-last_checkpoint):
			start_time = time.time()
			feed_dict = { self.curr_step : step}

			if (step) % self.conf.save_interval == 0:
				loss_value, images, labels, preds, summary, _ = self.sess.run(
					[self.reduced_loss,
					self.image_batch,
					self.label_batch,
					self.pred,
					self.total_summary,
					self.train_op],
					feed_dict=feed_dict)
				self.summary_writer.add_summary(summary, step+last_checkpoint)
				self.save(self.saver, step+last_checkpoint)
			else:
				loss_value, _ = self.sess.run([self.reduced_loss, self.train_op],
					feed_dict=feed_dict)

			duration = time.time() - start_time
			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step+last_checkpoint, loss_value, duration))
			write_log('{:d}, {:.3f}'.format(step+last_checkpoint, loss_value), self.conf.logfile)

			#write logs for tensorboard
			summary2 = self.sess.run(write_op, {log_var: loss_value})
			self.summary_writer.add_summary(summary2, step+last_checkpoint)
			#self.summary_writer.flush()
		

		# finish
		self.coord.request_stop()
		self.coord.join(threads)

	# evaluate
	def test(self):
		self.test_setup()

		self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.local_variables_initializer())

		# load checkpoint
		checkpointfile = self.conf.modeldir+ '/model.ckpt-' + str(self.conf.valid_step)
		self.load(self.loader, checkpointfile)

		# Start queue threads.
		threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

		areasOverlap = np.zeros(self.conf.valid_num_steps, np.float32)
		areasPredicted = np.zeros(self.conf.valid_num_steps, np.float32)
		areasGtObj = np.zeros(self.conf.valid_num_steps, np.float32)
        
		# Test!
		for step in range(self.conf.valid_num_steps):
			preds, _, _, areaOverlap, areaGTObj, areaPredicted, conv_out, conv_weights = self.sess.run([self.pred, self.accu_update_op, self.mIou_update_op, self.areaOverlap, self.areaGTObj, self.areaPredicted, tf.get_collection('conv_output'),  tf.get_collection('conv_weights')])
            
			if(self.conf.create_plots):
				print('create plot '+ plot_name)
				self.plot_conv_output(conv_out[i], plot_name)
				'''
				for i in range(len(conv_out)):
					plot_name='step_{}_conv_{}'.format(step, i)
					print('create plot '+ plot_name)
					self.plot_conv_output(conv_out[i], plot_name)
				for i in range(len(conv_weights)):
					plot_name='step_{}_weights_{}'.format(step, i)
					print('create plot '+ plot_name)
					self.plot_conv_weights(conv_weights[i], plot_name)
				'''

            #Save conv imaages
			areasOverlap[step] = areaOverlap
			areasGtObj[step] = areaGTObj
			areasPredicted[step] = areaPredicted
            
			print('step {:d}'.format(step))
			print('\tareaOverlap: {:.0f}'.format(areaOverlap))
			print('\tareaGTObj: {:.0f}'.format(areaGTObj))
			print('\tareaPredicted: {:.0f}'.format(areaPredicted))
          
		print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=self.sess)))
		print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=self.sess)))

		tp = ((areasOverlap[areasGtObj>0.0] / areasGtObj[areasGtObj>0]) >= 0.5).sum()
		fp = ((areasOverlap[areasGtObj>0.0] / areasGtObj[areasGtObj>0]) < 0.5).sum()
		tn = (areasGtObj[areasPredicted <= 0] <= 0).sum()
		fn = (areasGtObj[areasPredicted > 0] <= 0).sum()
       
		print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)
		precision = tp/(tp + fp);
		recall = tp/(tp + fn);
		score = (2*precision*recall)/(precision+recall);
        
		print('precision', precision, 'recall', recall, 'score', score)
        
		self.coord.request_stop()
		self.coord.join(threads)

	def train_setup(self):
		tf.set_random_seed(self.conf.random_seed)
		
		# Create queue coordinator.
		self.coord = tf.train.Coordinator()

		# Input size
		input_size = (self.conf.input_height, self.conf.input_width)
		
		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.data_list,
				input_size,
				self.conf.random_scale,
				self.conf.random_mirror,
				self.conf.ignore_label,
				self.coord)
			self.image_batch, self.label_batch = reader.dequeue(self.conf.batch_size)
		self.image_batch = tf.identity( self.image_batch, name='input_batch' )
		self.image_batch -= IMG_MEAN
		# Create network
		if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50, or deeplab")
			sys.exit(-1)
		elif self.conf.encoder_name == 'deeplab':
			net = Deeplab_v2(self.image_batch, self.conf.num_classes, True)
			# Variables that load from pre-trained model.
			restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
			# Trainable Variables
			all_trainable = tf.trainable_variables()
			# Fine-tune part
			encoder_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
			# Decoder part
			decoder_trainable = [v for v in all_trainable if 'fc' in v.name]
		else:
			net = ResNet_segmentation(self.image_batch, self.conf.num_classes, True, self.conf.encoder_name)
			# Variables that load from pre-trained model.
			restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name]
			# Trainable Variables
			all_trainable = tf.trainable_variables()
			# Fine-tune part
			encoder_trainable = [v for v in all_trainable if 'resnet_v1' in v.name] # lr * 1.0
			# Decoder part
			decoder_trainable = [v for v in all_trainable if 'decoder' in v.name]
					
		decoder_w_trainable = [v for v in decoder_trainable if 'weights' in v.name or 'gamma' in v.name] # lr * 10.0
		decoder_b_trainable = [v for v in decoder_trainable if 'biases' in v.name or 'beta' in v.name] # lr * 20.0
		# Check
		assert(len(all_trainable) == len(decoder_trainable) + len(encoder_trainable))
		assert(len(decoder_trainable) == len(decoder_w_trainable) + len(decoder_b_trainable))

		# Network raw output
		raw_output = net.outputs # [batch_size, h, w, 21]

		# Output size
		output_shape = tf.shape(raw_output)
		output_size = (output_shape[1], output_shape[2])

		# Groud Truth: ignoring all labels greater or equal than n_classes
		label_proc = prepare_label(self.label_batch, output_size, num_classes=self.conf.num_classes, one_hot=False)
		raw_gt = tf.reshape(label_proc, [-1,])
		indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
		gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
		raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
		prediction = tf.gather(raw_prediction, indices)

		# Pixel-wise softmax_cross_entropy loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
		# L2 regularization
		l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in all_trainable if 'weights' in v.name]
		# Loss function
		self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

		# Define optimizers
		# 'poly' learning rate
		base_lr = tf.constant(self.conf.learning_rate)
		self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
		learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.conf.num_steps), self.conf.power))
		# We have several optimizers here in order to handle the different lr_mult
		# which is a kind of parameters in Caffe. This controls the actual lr for each
		# layer.
		opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.conf.momentum)
		opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.conf.momentum)
		opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.conf.momentum)
		# To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
		# Instead, we separate the steps compute_grads+update_params.
		# Compute grads
		grads = tf.gradients(self.reduced_loss, encoder_trainable + decoder_w_trainable + decoder_b_trainable)
		grads_encoder = grads[:len(encoder_trainable)]
		grads_decoder_w = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_w_trainable))]
		grads_decoder_b = grads[(len(encoder_trainable) + len(decoder_w_trainable)):]
		# Update params
		train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable))
		train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, decoder_w_trainable))
		train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, decoder_b_trainable))
		# Finally, get the train_op!
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for collecting moving_mean and moving_variance
		with tf.control_dependencies(update_ops):
			self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

		# Saver for storing checkpoints of the model
		self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

		# Loader for loading the pre-trained model
		self.loader = tf.train.Saver(var_list=restore_var)

		# Training summary
		# Processed predictions: for visualisation.
		raw_output_up = tf.image.resize_bilinear(raw_output, input_size)
		raw_output_up = tf.argmax(raw_output_up, axis=3)
		self.pred = tf.expand_dims(raw_output_up, dim=3)
		# Image summary.
		images_summary = tf.py_func(inv_preprocess, [self.image_batch, 2, IMG_MEAN], tf.uint8)
		labels_summary = tf.py_func(decode_labels, [self.label_batch, 2, self.conf.num_classes], tf.uint8)
		preds_summary = tf.py_func(decode_labels, [self.pred, 2, self.conf.num_classes], tf.uint8)
		self.total_summary = tf.summary.image('images',
			tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
			max_outputs=2) # Concatenate row-wise.
		if not os.path.exists(self.conf.logdir):
			os.makedirs(self.conf.logdir)
		self.summary_writer = tf.summary.FileWriter(self.conf.logdir, graph=tf.get_default_graph())

	def test_setup(self):
		# Create queue coordinator.
		self.coord = tf.train.Coordinator()

		# Load reader
		with tf.name_scope("create_inputs"):
			reader = ImageReader(
				self.conf.data_dir,
				self.conf.valid_data_list,
				None, # the images have different sizes
				False, # no data-aug
				False, # no data-aug
				self.conf.ignore_label,
				self.coord)
			image, label = reader.image, reader.label # [h, w, 3 or 1]
		# Add one batch dimension [1, h, w, 3 or 1]
		self.image_batch, self.label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)
		self.image_batch = tf.identity( self.image_batch, name='image_batch')
		self.image_batch -= IMG_MEAN
		# Create network
		if self.conf.encoder_name not in ['res101', 'res50', 'deeplab']:
			print('encoder_name ERROR!')
			print("Please input: res101, res50, or deeplab")
			sys.exit(-1)
		elif self.conf.encoder_name == 'deeplab':
			net = Deeplab_v2(self.image_batch, self.conf.num_classes, False)
		else:
			net = ResNet_segmentation(self.image_batch, self.conf.num_classes, False, self.conf.encoder_name)
		
		# predictions
		raw_output = net.outputs
		raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.image_batch)[1:3,])
		raw_output = tf.argmax(raw_output, axis=3)
		pred = tf.expand_dims(raw_output, dim=3)
		self.pred = tf.reshape(pred, [-1,], name="predictions")
		# labels
		gt = tf.reshape(self.label_batch, [-1,])
		# Ignoring all labels greater than or equal to n_classes.
		temp = tf.less_equal(gt, self.conf.num_classes - 1)
		weights = tf.cast(temp, tf.int32)

		# fix for tf 1.3.0
		gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

		# Pixel accuracy
		self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
			self.pred, gt, weights=weights)

		# mIoU
		self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
			self.pred, gt, num_classes=self.conf.num_classes, weights=weights)

		# f1 score
		pred = tf.cast(self.pred, tf.int32)
		gt = tf.cast(gt, tf.int32)
        
		self.areaOverlap = tf.count_nonzero(pred * gt)
		self.areaGTObj = tf.count_nonzero(gt)
		self.areaPredicted = tf.count_nonzero(pred)
        
        
		# Loader for loading the checkpoint
		self.loader = tf.train.Saver(var_list=tf.global_variables())

	def save(self, saver, step):
		'''
		Save weights.
		'''
		model_name = 'model.ckpt'
		checkpoint_path = os.path.join(self.conf.modeldir, model_name)
		if not os.path.exists(self.conf.modeldir):
			os.makedirs(self.conf.modeldir)
		saver.save(self.sess, checkpoint_path, global_step=step)
		print('The checkpoint has been created.')
		#se abilitato invio messaggio su telegram
		if self.conf.telegram !="none":
			notifica_telegram.invio("TRAINING",checkpoint_path,step,(self.conf.num_steps+1),self.conf.telegram )

	def load(self, saver, filename):
		'''
		Load trained weights.
		''' 
		saver.restore(self.sess, filename)
		print("Restored model parameters from {}".format(filename))
        
	def plot_conv_weights(self, weights, name, channels_all=True):
		plot_dir = os.path.join('./plots', 'conv_weights')
		plot_dir = os.path.join(plot_dir, name)
		# create directory if does not exist, otherwise empty it
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		w_min = np.min(weights)
		w_max = np.max(weights)
		channels = [0]
		if channels_all:
			channels = range(weights.shape[2])
		num_filters = weights.shape[3]
		grid_r, grid_c = self.get_grid_dim(num_filters)
		fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]))
		# iterate channels
		for channel in channels:
			# iterate filters inside every channel
			for l, ax in enumerate(axes.flat):
				# get a single filter
				img = weights[:, :, channel, l]
				# put it on the grid
				ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
				# remove any labels from the axes
				ax.set_xticks([])
				ax.set_yticks([])
		plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')
		plt.close(fig)
        
	def plot_conv_output(self, conv_img, name):
        # make path to output folder
		plot_dir = os.path.join('./plots', 'conv_output')
		plot_dir = os.path.join(plot_dir, name)
        # create directory if does not exist, otherwise empty it
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		w_min = np.min(conv_img)
		w_max = np.max(conv_img)
        # get number of convolutional filters
		num_filters = conv_img.shape[3]
        # get number of grid rows and columns
		grid_r, grid_c = self.get_grid_dim(num_filters)
        # create figure and axes
		fig, axes = plt.subplots(min([grid_r, grid_c]), max([grid_r, grid_c]), figsize=(150, 150))
        # iterate filters
		for l, ax in enumerate(axes.flat):
			# get a single image
			img = conv_img[0, :, :,  l]
            # put it on the grid
			ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
            # remove any labels from the axes
			ax.set_xticks([])
			ax.set_yticks([])
        # save figure
		plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
		plt.close(fig)
        
	def get_grid_dim(self, x):
		factors = self.prime_powers(x)
		if len(factors) % 2 == 0:
			i = int(len(factors) / 2)
			return factors[i], factors[i - 1]
		i = len(factors) // 2
		return factors[i], factors[i]
    
	def prime_powers(self, n):
		factors = set()
		for x in range(1, int(math.sqrt(n)) + 1):
			if n % x == 0:
				factors.add(int(x))
				factors.add(int(n // x))
		return sorted(factors)
