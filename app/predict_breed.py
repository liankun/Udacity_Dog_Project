import argparse
import numpy as np
from glob import glob
from Utility import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('input',
              action='store',
              help='image path')
parser.add_argument('--checkpoint',
              action='store',
              default='../saved_models',
              help='model checkpoint')
parser.add_argument('--training',
              action='store_true',
              default=False,
              help='retrain the model,need train,test images files or bottleneck features path')
parser.add_argument('--files_path',
              action='store',
              default='../dogImages',
              help='files path to train the model')
parser.add_argument('--bottleneck_feature_path',
              action='store',
              default='../bottleneck_features',
              help='bottleneck feature path')
parser.add_argument('--face_detector',
              action='store',
              default='cv2',
              help='choose one of the three face detectors: cv2 hog cnn')
parser.add_argument('--arch',
              action='store',
              default='Xception',
              help='choose one of the four network arch: VGG19 ResNet50 Inception Xception')
parser.add_argument('--augmentation',
              action='store_true',
              default=False,
              help='Use augmentation to train the model, warning: bottleneck features will not be used')
parser.add_argument('--use_bottleneck_feature',
              action='store_false',
              default=True,
              help='Use bottleneck features,False means the image will go from the beginning of the network')
parser.add_argument('--hidden_layer_nodes',
              action='store',
              default=128,
              help='number of nodes for the hidden layers',
              type=int)
parser.add_argument('--free_first_layers',
              action='store',
              default=0,
              help='free parameters of the first n layers for training',
              type=int)
parser.add_argument('--free_last_layers',
              action='store',
              default=0,
              help='free parameters of the last n layers for training',
              type=int)
parser.add_argument('--verbose',action='store_true',default=False)

result = parser.parse_args()
if result.verbose:
    print("image path:           {!r}".format(result.input))
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
    

train_files, train_targets = load_dataset(result.files_path+'/train')
valid_files, valid_targets = load_dataset(result.files_path+'/valid')
test_files, test_targets = load_dataset(result.files_path+'/test')
    
    # load list of dog names
dog_names = [item[20:-1] for item in sorted(glob(result.files_path+"/train/*/"))]
    # print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

