import argparse

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
parser.add_argument('--train_file_path',
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
    print("train file path :       {!r}".format(result.train_file_path))
    print("bottleneck features path:  {!r}".format(result.bottleneck_feature_path))
    print("face detector:         {!r}".format(result.face_detector))
    print("arch:               {!r}".format(result.arch))
    print("augmentation:          {!r}".format(result.augmentation))
    print("use bottleneck feature:   {!r}".format(result.use_bottleneck_feature))
    print("hidden layer nodes:      {!r}".format(result.hidden_layer_nodes))
    print("free first n layers:     {!r}".format(result.free_first_layers))
    print("free last n layers:      {!r}".format(result.free_last_layers))
    
