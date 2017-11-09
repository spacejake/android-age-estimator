# Preparing a TF model for usage in Android
# By Omid Alemi - Jan 2017
# Works with TF r1.0

import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = 'age-model.pbtxt' # Path to pbtext
checkpoint_path = 'checkpoint-14999' # Checkpoint to load (extensions auto loaded)
input_saver_def_path = ""
input_binary = False
output_node_names = "output/output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen-age-model.pb'
output_optimized_graph_name = 'optimized-age-model.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

#images = tf.placeholder(tf.float32, [None, 227, 227, 3])

    writer = tf.summary.FileWriter("/tmp/age_tf_log/...", input_graph_def)


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        #["batch_processing/Reshape"], # an array of the input node(s)
        #["batch_processing/ParseSingleExample"]
        #["Placeholder:0"],
        ["input"], # an array of the input node(s)
        ["output/output"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())

#writer = tf.summary.FileWriter("/tmp/age_tf_log/...", output_graph_def)

#tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)                    
