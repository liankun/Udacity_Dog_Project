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
parser.add_argument('--epochs',
              action='store',
              default=10,
              help="number of epoch for training",
              type=int)
parser.add_argument('--best_model',
              action='store',
              default=config_info['best_model'],
              help='the name of best model, the name configure is (weights.best.Xception.128.0.0.Bottleneck.hdf5 for example):\
                   arch: Xception\
                   hidden layer nodes:128\
                   free first n layers:0\
                   free last n layers:0\
                   Full network or bottleneck: Bottleneck')
parser.add_argument('--create_pickle_files',
              action='store_true',
              default=False,
              help="create pickle files for training,valid,test images pathes and dog names.This should be done when files path changed"
              )

parser.add_argument('--verbose',action='store_true',default=False)

result = parser.parse_args()

if result.arch not in ['VGG19','ResNet50','Inception','Xception']:
    print("Bad architecture and will set to Xception")
    result.arch = 'Xception'

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
    print("number of epochs:       {!r}".format(result.epochs))
    print("best model name:        {!r}".format(result.best_model))
    print("create pickle files:     {!r}".format(result.create_pickle_files))
    

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



if result.create_pickle_files:    
    train_files, train_targets = load_dataset(result.files_path+'/train')
    valid_files, valid_targets = load_dataset(result.files_path+'/valid')
    test_files, test_targets = load_dataset(result.files_path+'/test')
    
    # list of dog names
    dog_names = [item[20:-1] for item in sorted(glob(result.files_path+"/train/*/"))]
    
    # print statistics about the dataset
    print('\n')
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))
    
    #save to pickle files
    with open("dog_names.pk",'wb') as fp:
        pickle.dump(dog_names,fp)
    
    #train files
    with open("train_files.pk",'wb') as fp:
        pickle.dump(train_files,fp)
    with open("train_targets.pk",'wb') as fp:
        pickle.dump(train_targets,fp)
    
    #valid files
    with open("valid_files.pk",'wb') as fp:
        pickle.dump(valid_files,fp)
    with open("valid_targets.pk",'wb') as fp:
        pickle.dump(valid_targets,fp)
    
    #test files
    with open("test_files.pk",'wb') as fp:
        pickle.dump(test_files,fp)
    with open("test_targets.pk",'wb') as fp:
        pickle.dump(test_targets,fp)
    
    


if not result.training:
    predict_breed(result.input,
             model_path=result.checkpoint+"/"+result.best_model,
             bottleneck_feature_path=result.bottleneck_feature_path,
             show_image=result.show_image,
             face_method=result.face_detector)

else:    
    # training one model
    new_model_name='weights.best.{}.{}.{}.{}.Bottleneck.hdf5'.format(result.arch,
                                               result.hidden_layer_nodes,
                                               result.free_first_layers,
                                               result.free_last_layers)
    if not result.use_bottleneck_feature:
        #for this case will use full network
        new_model_name='weights.best.{}.{}.{}.{}.FullNetWork.hdf5'.format(result.arch,
                                               result.hidden_layer_nodes,
                                               result.free_first_layers,
                                               result.free_last_layers)
    print(new_model_name)
    train_model(btnk_path=result.bottleneck_feature_path,
            new_model_path=result.checkpoint+"/"+new_model_name,
            epochs=result.epochs,
            hidden_nodes=result.hidden_layer_nodes,
            arch = result.arch,
            use_btnk = result.use_bottleneck_feature,
            free_first_nlayers=result.free_first_layers,
            free_last_nlayers=result.free_last_layers,
            augmentation = result.augmentation
            )
    config_info['best_model']=new_model_name



# write the new config files to disk
dict_json = json.dumps(config_info)
with open("config.json","w") as f:
    f.write(dict_json)    
        
    
    