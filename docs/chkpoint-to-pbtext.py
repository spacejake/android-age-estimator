# This file is based on source from https://github.com/dpressel/rude-carnie
# Source was refined and Isolated to specifically generate an android compatible model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import os
import json
import csv
import re
from tensorflow.contrib.layers import *

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

TOWER_NAME = 'tower'
RESIZE_FINAL = 227
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('output_model_dir', './', 'directory to output model')
tf.app.flags.DEFINE_string('output_model_name', 'age-model.pbtxt', 'name of model to be outputed')


FLAGS = tf.app.flags.FLAGS

def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inception_v3(nlabels, images, pkeep, is_training):

    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
    weight_decay = 0.00004
    stddev=0.1
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:

        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = tf.nn.dropout(net, pkeep, name='droplast')
                    net = flatten(net, scope="flatten")
    
    with tf.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(net, weights), biases, name=scope.name)
        _activation_summary(output)
    return output

def main(argv=None):  # pylint: disable=unused-argument

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        nlabels = len(AGE_LIST)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = inception_v3

        with tf.device(FLAGS.device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3], name="input")
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
            
            requested_step = FLAGS.requested_step if FLAGS.requested_step else None
        
            checkpoint_path = '%s' % (FLAGS.model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
            
            graph = sess.graph

            tensors = tf.contrib.graph_editor.get_tensors(graph)
            #print("Tensor Names: {}".format(tensors))

            # `sess.graph` provides access to the graph used in a `tf.Session`.
            # See graph structure in tensorboard, tensorboard --logdir "/tmp/age_chk_log/" 
            writer = tf.summary.FileWriter("/tmp/age_chk_log/...", graph)


            #print([node.name for node in graph.as_graph_def().node])
            #print("logits: {}".format(logits))

            softmax_output = tf.nn.softmax(logits, name="output")
            #print(" tf.nn: {}".format( tf.nn))

            #print("softmax_output: {}".format(softmax_output))

            tf.train.write_graph(sess.graph_def, FLAGS.output_model_dir, FLAGS.output_model_name, as_text=True)
            return

if __name__ == '__main__':
    tf.app.run()
