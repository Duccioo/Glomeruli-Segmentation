import argparse
import os
import tensorflow as tf
from model import Model
import sys
import glob




#definisco da linea di comando a quale fold mi sto riferendo
#esemprio fold='/deeplearn/F1/'
parser = argparse.ArgumentParser()
parser.add_argument('--folder', dest='folder', type=str, default='train')

#indica il nomero al quale salvare il checkpoint
parser.add_argument("-c",'--checkpoint', dest='checkpoint', type=int, default='train')

#indica se voglio ricevere una notifica su telegram ogni qual volta faccio il 25% del training
parser.add_argument("-T",'--telegram', action="store_true")

#indica la cartella dove sono le immagini preprocessate
parser.add_argument('--data_preprocessed', dest='data_preprocessed', type=str, default='train')
args = parser.parse_args()
fold=args.folder



last_checkpoint=0

#controllo se nella cartella model del folder è già presente un checkpoint:
model_dir=glob.glob(fold+"/model/*.index")
if model_dir != []:
	#se è presente almeno un file va a prendere quello con il numero maggiore
	#il formato del nome del modello checkpoint: 'model.ckpt-'+'numero' es: model.ckpt-2060.data-00000-of-00001
	sorted_model_dir=sorted(model_dir,key=lambda x: int(x.split('-')[1].replace(".index","")),reverse=True)
	checkpoint=sorted_model_dir[0].replace(".index","")
	try:
		checkpoint2=sorted_model_dir[1].replace(".index","")
	except:
		checkpoint2="error"


	last_checkpoint=int(checkpoint.split('-')[1])
	print(last_checkpoint)
	if last_checkpoint==0:
		checkpoint='./PretrainResnet/resnet_v1_101.ckpt'
else:
	#altrimenti se non c'è nessun checkpoint vecchio allora prende quello di default
	checkpoint='./PretrainResnet/resnet_v1_101.ckpt'

print("CHECKPOINT:",checkpoint)
#per ottimizzare lo spazio occupato sul drive rimuovo i file più vecchi rispetto al checkpoint più nuovo
for i in glob.glob(fold+"/model/*"):
	if i!=(checkpoint+".data-00000-of-00001") and i!=(checkpoint+".index") and i!=(checkpoint+".meta") and i!=(checkpoint2+".data-00000-of-00001") and i!=(checkpoint2+".index") and i!=(checkpoint2+".meta") and i!="checkpoint":
		os.remove(i)
		print("rimozione file",i)
	

#conto il numero di righe nel file
train=fold+"/train.txt"
file = open(train, "r")
nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
n_righe = len(nonempty_lines)

#----------------------------------------------#CONFIG:#---------------------------------------------------------#
split_data= n_righe #numero di immagini di train totali
epoch=10 #numero di epoche
power=0.99 #potenza
batch_size=2
num_steps=(int(split_data/batch_size)*epoch)
save_inverval=args.checkpoint
data_dir=args.data_preprocessed
if args.telegram:
	telegram_chat="-1001541977258"
else:
	telegram_chat="none"

print(num_steps)

parser.add_argument('--option', dest='option', type=str, default='train', help='actions: train, test, or predict')
args = parser.parse_args()
	
def configure():
	flags = tf.app.flags

	# training
	flags.DEFINE_integer('num_steps',num_steps , 'maximum number of iterations')
	flags.DEFINE_integer('save_interval',save_inverval, 'number of iterations for saving and visualization')
	flags.DEFINE_integer('random_seed', 1234, 'random seed')
	flags.DEFINE_float('weight_decay', 0.0005, 'weight decay rate')
	flags.DEFINE_float('learning_rate', 2.5e-4, 'learning rate')
	flags.DEFINE_float('power', power, 'hyperparameter for poly learning rate')
	flags.DEFINE_float('momentum', 0.9, 'momentum')
	flags.DEFINE_string('encoder_name', 'res101', 'name of pre-trained model, res101, res50 or deeplab')
	flags.DEFINE_string('pretrain_file', checkpoint, 'pre-trained model filename corresponding to encoder_name')
	flags.DEFINE_string('data_list', fold+'/train.txt', 'training data list filename')
	#--aggiunte da me--#
	flags.DEFINE_integer('last_checkpoint',last_checkpoint,'ultimo checkpoint salvato da cui ripartire')

	if args.telegram:
		flags.DEFINE_integer('telegram',telegram_chat,'id della chat a cui inviare i messagi su telegram')
	
	# testing / validation
	flags.DEFINE_integer('valid_step', 2000, 'checkpoint number for testing/validation')
	flags.DEFINE_integer('valid_num_steps', 3718, '= number of testing/validation samples')
	flags.DEFINE_string('valid_data_list', './dataset/valid.txt', 'testing/validation data list filename')
	flags.DEFINE_boolean('create_plots', False, 'whether to create plots')

	# data
	flags.DEFINE_string('data_dir', data_dir, 'data directory')
	flags.DEFINE_integer('batch_size', 2, 'training batch size')
	flags.DEFINE_integer('input_height', 512, 'input image height')
	flags.DEFINE_integer('input_width', 512, 'input image width')
	flags.DEFINE_integer('num_classes', 2, 'number of classes')
	flags.DEFINE_integer('ignore_label', 0, 'label pixel value that should be ignored')
	flags.DEFINE_boolean('random_scale', False, 'whether to perform random scaling data-augmentation')
	flags.DEFINE_boolean('random_mirror', False, 'whether to perform random left-right flipping data-augmentation')
	
	# log
	flags.DEFINE_string('modeldir', fold+'/model', 'model directory')
	flags.DEFINE_string('logfile', 'log.txt', 'training log filename')
	flags.DEFINE_string('logdir', 'log', 'training log directory')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS

def main(_):
	
		# Set up tf session and initialize variables. 
		# config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		# sess = tf.Session(config=config)
	sess = tf.Session()
		# Run
	model = Model(sess, configure())
	getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
	tf.app.run()