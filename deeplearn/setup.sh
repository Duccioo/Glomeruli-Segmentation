#Resnet Model herunterladen
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz -P ./PretrainResnet
tar -xvf ./PretrainResnet/resnet_v1_101_2016_08_28.tar.gz -C ./PretrainResnet

#Ordner für Inputdaten anlegen
mkdir data

#Ordner für das Preprocessing anlegen
mkdir data_preprocessed
mkdir data_preprocessed/masks
mkdir data_preprocessed/img