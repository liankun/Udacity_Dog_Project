import argparse
import json
import ast

import numpy as np
from glob import glob
from Utility import *


import pickle

#read the config files for default setting
#config_info = {}
with open('config.json') as config_json:
    config_info = json.load(config_json)


#print(config_info)

parser = argparse.ArgumentParser()

parser.add_argument('--input',
              action='store',
              default=config_info['input'],
              help='image path for predict')
parser.add_argument("--show_image",
              action='store_true',
              default=False,
              help="show most similar image and original image")
parser.add_argument('--checkpoint',
              action='store',
              default=config_info['checkpoint'],
              help='model checkpoint')
parser.add_argument('--training',
              action='store_true',
              default=False,
              help='retrain the model,need train,test images files or bottleneck features path')
parser.add_argument('--files_path',
              action='store',
              default=config_info['files_path'],
              help='files path to train the model')
parser.add_argument('--bottleneck_feature_path',
              action='store',
              default=config_info['bottleneck_feature_path'],
              help='bottleneck feature path')
parser.add_argument('--face_detector',
              action='store',
              default=config_info['face_detector'],
              help='choose one of the three face detectors: cv2 hog cnn')
parser.add_argument('--arch',
              action='store',
              default=config_info['arch'],
              help='choose one of the four network arch: VGG19 ResNet50 Inception Xception')
parser.add_argument('--augmentation',
              action='store',
              default=config_info['augmentation'],
              help='Use augmentation to train the model, warning: use_bottleneck_feature need to be set to False ',
              type=ast.literal_eval)
parser.add_argument('--use_bottleneck_feature',
              action='store',
              default=config_info['use_bottleneck_feature'],
              help='Use bottleneck features,False means the image will go from the beginning of the network.\
                  set this to False will disable free_first_layers ,free_last_layers and augmentation',
              type=ast.literal_eval)
parser.add_argument('--hidden_layer_nodes',
              action='store',
              default=config_info['hidden_layer_nodes'],
              help='number of nodes for the hidden layers',
              type=int)
parser.add_argument('--free_first_layers',
              action='store',
              default=config_info['free_first_layers'],
              help='free parameters of the first n layers for training,\
                  use_bottleneck_feature need to be set to False',
              type=int)
parser.add_argument('--free_last_layers',
              action='store',
              default=config_info['free_last_layers'],
              help='free parameters of the last n layers for training,\
                  use_bottleneck_feautre need to be set to False',
              type=int)
parser.add_argument('--epoch',
              action='store',
              default=10,
              help="number of epoch for training",
              type=int)
parser.add_argument('--continue',
              action='store_true',
              default=False,
              help='continue training the model from last best model checkpoint')
parser.add_argument('--best_model',
              action='store',
              default=config_info['best_model'],
              help='the name of best model, the name configure is (weights.best.Xception.128.0.0.Bottleneck.hdf5 for example):\
                   arch: Xception\
                   hidden layer nodes:128\
                   free first n layers:0\
                   free last n layers:0\
                   Full network or bottleneck: Bottleneck')
parser.add_argument('--verbose',action='store_true',default=False)

result = parser.parse_args()
if result.verbose:
    print("image path:           {!r}".format(result.input))
    print("show image:           {!r}".format(result.show_image))
    print("checkpoint:           {!r}".format(result.checkpoint))
    print("training :            {!r}".format(result.training))
    print("train file path :       {!r}".format(result.files_path))
    print("bottleneck features path:  {!r}".format(result.bottleneck_feature_path))
    print("face detector:         {!r}".format(result.face_detector))
    print("arch:               {!r}".format(result.arch))
    print("augmentation:          {!r}".format(result.augmentation))
    print("use bottleneck feature:   {!r}".format(result.use_bottleneck_feature))
    print("hidden layer nodes:      {!r}".format(result.hidden_layer_nodes))
    print("free first n layers:     {!r}".format(result.free_first_layers))
    print("free last n layers:      {!r}".format(result.free_last_layers))
    print("best model name:        {!r}".format(result.best_model))
    

config_info['input'] = result.input
config_info['checkpoint'] = result.checkpoint
config_info['files_path'] = result.files_path
config_info['bottleneck_feature_path'] = result.bottleneck_feature_path
config_info['face_detector'] = result.face_detector
config_info['arch'] = result.arch
config_info['augmentation'] = result.augmentation
config_info['use_bottleneck_feature'] = result.use_bottleneck_feature
config_info['hidden_layer_nodes'] = result.hidden_layer_nodes
config_info['free_first_layers'] = result.free_first_layers
config_info['free_last_layers'] = result.free_last_layers
config_info['best_model'] = result.best_model


# write the new config files to disk
dict_json = json.dumps(config_info)
with open("config.json","w") as f:
    f.write(dict_json)
    
#train_files, train_targets = load_dataset(result.files_path+'/train')
#valid_files, valid_targets = load_dataset(result.files_path+'/valid')
#test_files, test_targets = load_dataset(result.files_path+'/test')


    
# load list of dog names
#dog_names = [item[20:-1] for item in sorted(glob(result.files_path+"/train/*/"))]
# print statistics about the dataset
#print('\n')
#print('There are %d total dog categories.' % len(dog_names))
#print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
#print('There are %d training dog images.' % len(train_files))
#print('There are %d validation dog images.' % len(valid_files))
#print('There are %d test dog images.'% len(test_files))

#with open("dog_names.pk",'wb') as fp:
#    pickle.dump(dog_names,fp)

#with open("dog_names.pk","rb") as fp:
#    dog_names = pickle.load(fp)
#print(dog_names)
    
#get bottleneck feature for wanted arch
#first is train, second is valid, last is test
#arch = config_info['arch']
#btnk_ftr = [{},{},{}]
#btnk_ftr[0][arch],btnk_ftr[1][arch],btnk_ftr[2][arch] = get_bottleneck_features(config_info['bottleneck_feature_path'],arch=arch)


#make prediction
#load_model(config_info['checkpoint']+"/"+config_info['best_model'])

#get_bottleneck_features(config_info['bottleneck_feature_path'],arch=config_info['arch'])

#print(face_detector(config_info['input'],config_info['face_detector']))

#print(dog_detector(config_info['input']))

if not result.training:
    predict_breed(result.input,
             model_path=result.checkpoint+"/"+result.best_model,
             bottleneck_feature_path=result.bottleneck_feature_path,
             files_path=result.files_path,
             show_image=result.show_image,
             face_method=result.face_detector)

else:
    #train_files, train_targets = load_dataset(result.files_path+'/train')
    #valid_files, valid_targets = load_dataset(result.files_path+'/valid')
    #test_files, test_targets = load_dataset(result.files_path+'/test')

    model = create_model(hidden_nodes=result.hidden_layer_nodes,
                  arch = result.arch,
                  use_btnk = result.use_bottleneck_feature,
                  free_first_nlayers=result.free_first_layers,
                  free_last_nlayers=result.free_last_layers
                   )
    if result.use_bottleneck_feature:
        btnk_train,btnk_valid,btnk_test=get_bottleneck_features(result.bottleneck_feature_path,arch=arch)
        model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
        
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', 
                               verbose=1, save_best_only=True)
        Xception_model.fit(train_Xception, train_targets, 
          validation_data=(valid_Xception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=2)

        
    
    