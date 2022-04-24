import model
import tensorflow as tf
import sys
import glob
tf.VERSION

#CONFIG
folder_in=sys.argv[1] #cartella dove prendere il modello
folder_out=sys.argv[2] #cartella dove salvarlo
data_dir= sys.argv[3] #cartella di dove sono le immagini

#controllo se nella cartella model del folder è già presente un checkpoint:
model_dir=glob.glob(folder_in+"/model/*.index")
if model_dir != []:
	#se è presente almeno un file va a prendere quello con il numero maggiore
	#il formato del nome del modello checkpoint: 'model.ckpt-'+'numero'
	sorted_model_dir=sorted(model_dir,reverse=True)
	checkpoint=sorted_model_dir[0].replace(".index","")
	last_checkpoint=int(checkpoint.split('-')[1])
else:
    last_checkpoint=0	
#------------------------------------------------------------------#

class Config:
    def __init__(self):
        self.valid_step = last_checkpoint
        self.valid_num_steps = 0
        self.valid_data_list = '/content/drive/MyDrive/ParallelCrossValidation/SETUP/valid.txt'
        self.data_dir = data_dir
        self.batch_size = 2
        self.input_height = 512
        self.input_width = 512
        self.num_classes = 2
        self.ignore_label = 0
        self.random_scale = False
        self.random_mirror = False
        self.modeldir = folder_in+'/model'
        self.logfile = 'log.txt'
        self.logdir = 'log'
        self.encoder_name = 'res101'


sess = tf.Session()
m = model.Model(sess, Config())
m.test_setup()

from tensorflow.python.framework import graph_util

output_graph = folder_out+"/frozen_model.pb"

# Before exporting our graph, we need to precise what is our output node
# This is how TF decides what part of the Graph he has to keep and what part it can dump
# NOTE: this variable is plural, because you can have multiple output nodes
input_node_names = "image_batch"
output_node_names = "predictions"

# We retrieve the protobuf graph definition
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

m.sess.run(tf.global_variables_initializer())
m.sess.run(tf.local_variables_initializer())

# load checkpoint
checkpointfile = m.conf.modeldir+ '/model.ckpt-' + str(m.conf.valid_step)
m.load(m.loader, checkpointfile)


# We use a built-in TF helper to export variables to constants
output_graph_def = graph_util.convert_variables_to_constants(
    sess, # The session is used to retrieve the weights
    input_graph_def, # The graph_def is used to retrieve the nodes
    output_node_names.split(",") # The output node names are used to select the usefull nodes
)

# Finally we serialize and dump the output graph to the filesystem
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
print("%d ops in the final graph." % len(output_graph_def.node))
print("Bravo testa di cazzo hai esportato il modello!!!!")

print("ADESSO PUOI PASSARE AL FILE DI TEST")

