from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint

import cv2
import face_recognition

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True    

import pickle

from extract_bottleneck_features import *


def load_dataset(path):
    """
    input: the path of image files
    return: dog_files and dog_targets
    
    dog_files: numpy array of dog image path
    dog_targets: numeric category for each dog image
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def get_bottleneck_features(path,arch='Xception'):
    """
    intput
    arch: the architecture of the network
    path: the path to the bottleneck features
    return
    a numpy array of train ,test and valid
    """
    file_name = None
    if arch == 'Xception':
        file_name = 'DogXceptionData.npz'
    elif arch == 'VGG19':
        file_name = 'DogVGG19Data.npz'
    elif arch == 'ResNet50':
        file_name = 'DogResnet50Data.npz'
    elif arch == 'Inception':
        file_name = 'DogInceptionV3Data.npz'
    else:
        #use default
        file_name = 'DogXceptionData.npz'
    
    bottleneck_features = np.load(path+'/'+file_name)
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    
    print("bottleneck train shape: ",train.shape)
    print("bottleneck valid shape: ",valid.shape)
    print("bottleneck test shape:  ",test.shape)
    return train,valid,test

def create_model(hidden_nodes=128,
           arch='Xception',
           use_btnk = True,
           free_first_nlayers=0,
           free_last_nlayers=0):
    """
    input
    hidden nodes: the number nodes for hidden layer
    arch: the arch used for transfer learning
    use_btnk: use bottleneck feature to train,if this set
    to True, free_first_nlayers and free_last_nlayers will
    be freezed
    free_first_nlayers: make the first n layers trainable 
    free_last_nlayers:make last n layers trainable
    """
    shape = (7,7,2048)
    if arch == 'VGG19':
        shape=(7,7,512)
    if arch =='ResNet50':
        shape=(1,1,2048)
    if arch == 'Inception':
        shape=(5,5,2048)
    
    #create model
    model = Sequential()
    if not use_btnk:
        if arch == 'VGG19':
            model.add(applications.VGG19(weights = 'imagenet',include_top=False,input_shape=(224,224,3)))
        elif arch =='ResNet50':
            model.add(applications.ResNet50(weights = 'imagenet',include_top=False,input_shape=(224,224,3)))
        elif arch == 'Inception':
            model.add(applications.InceptionV3(weights = 'imagenet',include_top=False,input_shape=(224,224,3)))
        else:
            #this is default
            model.add(applications.Xception(weights = 'imagenet',include_top=False,input_shape=(224,224,3)))
        model.add(GlobalAveragePooling2D())
        base_model = model.layers[0]
        #make them untrainable
        #base_model.trainable=False
        arch_depth = len(base_model.layers)
        print("depth of the arch: ",arch_depth)
        for layer_i in range(arch_depth):
            #set the first n layers trainable
            if layer_i<free_first_nlayers:
                continue
            #set last n layers trainable
            if layer_i>=arch_depth-free_last_nlayers:
                continue
            base_model.layers[layer_i].trainable = False         
    else:
        model.add(GlobalAveragePooling2D(input_shape=shape))
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(133,activation='softmax'))
         
    model.summary()
    return model


    
def load_model(path):
    """
    input: the path of model path and name
    return: 
    a model used to predict
    arch: the arch of the model
    use_bottleneck: use bottleneck feature or not
    a model will be created according to the 
    configure in the model name
    """
    model_name = path.split("/")[-1]
    model_config = model_name.split(".")
    arch = model_config[2]
    hidden_nodes = int(model_config[3])
    first_nlayers = int(model_config[4])
    last_nlayers = int(model_config[5])
    use_bottleneck = False
    if model_config[6] =="Bottleneck":
        use_bottleneck = True
    
    print("model arch: ",arch)
    print("hidden_nodes: ",hidden_nodes)
    print("first_nlayers: ",first_nlayers)
    print("last_nlayers: ",last_nlayers)
    print("use bottleneck: ",use_bottleneck)
    
    model = create_model(hidden_nodes=hidden_nodes,
                  arch = arch,
                  use_btnk = use_bottleneck,
                  free_first_nlayers=first_nlayers,
                  free_last_nlayers=last_nlayers
                   )
    
    model.load_weights(path)
    return model,arch,use_bottleneck


def face_detector(img_path,method='cv2'):
    """
    input
    img_path: image path
    method: method to detect human faces
    
    return
    True: human face detected
    False: human face not detected
    """
    print("running face detector")
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    if method !='cv2':
        tmp_location = face_recognition.face_locations(gray,number_of_times_to_upsample=0,model=method)
        return len(tmp_location)>0
        
    
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    
    return len(faces) > 0



def path_to_tensor(img_path):
    """
    input
    img_path: single image path
    return
    numpy array of the image
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """
    input
    img_paths: a list of image paths
    return
    an array of numpy of these image the first dimension is the number of
    the images
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



def dog_detector(img_path):
    """
    input
    img_path: the path for image
    return
    True: dog detected
    False: dog not detected
    """
    print("running dog_detector")
    #img = cv2.imread(img_path)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # define ResNet50 model for dog detector
    ResNet50_model = ResNet50(weights='imagenet')

    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img))
    return ((prediction <= 268) & (prediction >= 151)) 
    
def predict_breed(img_path,model_path,bottleneck_feature_path,show_image=False,face_method='cv2'):
    """
    input
    img_path: the path of image
    bottleneck_feature_path: path of bottleneck features
    files_path: path of train, validation and test
    show_image: show the image 
    face_method: face detector method
    model
    the path to model used to predict the breed
    return
    if dog or human detected, will return the closest breed
    if no dog or human detected, will return None
    """
    
    #check if dog or human exist or not
    contain_dog = dog_detector(img_path)
    contain_human = face_detector(img_path,face_method)
    
    if (not contain_dog) and (not contain_human):
        print("No Dog or human detected !")
        return
    if contain_dog:
        print("dog detected")
    if contain_human:
        print("human detected")
    
    model,arch,use_bottleneck = load_model(model_path)
    
    #choose the right architecture
    best_one = None
    if use_bottleneck:
        bottleneck_feature = None
        if arch == 'Xception':
            bottleneck_feature = extract_Xception(path_to_tensor(img_path))
        if arch == 'VGG19':
            bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
        if arch == 'ResNet50':
            bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
            #the new bottleneck feature dimension is dfferent
            bottleneck_feature = np.reshape(bottleneck_feature,(1,1,1,2048))
        if arch == 'Inception':
            bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))

        predicted_vector = model.predict(bottleneck_feature)
        best_one = np.argmax(predicted_vector)
    else:
        predicted_vector = model.predict(path_to_tensor(img_path).astype('float32')/255)
        best_one = np.argmax(predicted_vector)
        
    #print("shape of bottleneck feature for iuput image")
    #print(bottleneck_feature.shape)
    # load list of dog names
    with open("dog_names.pk","rb") as fp:
        dog_names = pickle.load(fp)
        
    print("most similar breed: ",dog_names[best_one])
    
    if not show_image:
        return dog_names[best_one]
    
    #load the train files
    #train_files, train_targets = load_dataset(files_path+'/train')
    with open("train_files.pk","rb") as fp:
        train_files = pickle.load(fp)
    
    with open("train_targets.pk","rb") as fp:
        train_targets = pickle.load(fp)
    
    
    #load the bottleneck features
    btnk_train,btnk_valid,btnk_test=get_bottleneck_features(bottleneck_feature_path,arch=arch)
    best_i = 0
    best_similar = -9999
    vect0 = bottleneck_feature.flatten()
    for i in range(btnk_train.shape[0]):
        #only consider the same group
        if np.argmax(train_targets[i])!=best_one:
            continue       
        vect1 = btnk_train[i].flatten()
        cosAngle = np.sum(vect0*vect1)/np.linalg.norm(vect0)/np.linalg.norm(vect1)
        
        if cosAngle>=best_similar:
            best_similar=cosAngle
            best_i = i
    
    #show image
    img0 = cv2.imread(img_path)
    img1 = cv2.imread(train_files[best_i])
    cv2.imshow('original image',img0)
    cv2.imshow('most similar image',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dog_names[best_one]
     
    
def train_model(btnk_path,
           new_model_path,
           epochs=10,
           hidden_nodes=128,
           arch='Xception',
           use_btnk = True,
           free_first_nlayers=0,
           free_last_nlayers=0,
           augmentation = False):
    """
    input
    btnk_path: path of bottleneck features
    new_model_path: the path+name of the new model want to train
    hidden_nodes: the number of nodes in hidden layers
    arch: the arch of the transfer learning
    use_btnk: use bottleneck feature to train or full network
    free_first_nlayers: make the first n layers trainable for seleted
    arch 
    free_last_nlayers: make the last n layers trainalble for seleted
    arch
    augmentation: whether to use augmentation technique
    continue_training: continue training based on last saved model
    return None
    This function will train the new model according to the requirement above
   
    """
    
    #get the data files
    with open("train_files.pk","rb") as fp:
        train_files = pickle.load(fp)
    with open("train_targets.pk","rb") as fp:
        train_targets = pickle.load(fp)
        
    with open("valid_files.pk","rb") as fp:
        valid_files = pickle.load(fp)
    with open("valid_targets.pk","rb") as fp:
        valid_targets = pickle.load(fp)
        
    with open("test_files.pk","rb") as fp:
        test_files = pickle.load(fp)
    with open("test_targets.pk","rb") as fp:
        test_targets = pickle.load(fp)
    
    #print("test_files shape ",test_files.shape)
    #print("test_targets shape ",test_targets.shape)
    
    #create model
    model = create_model(hidden_nodes=hidden_nodes,
                  arch = arch,
                  use_btnk = use_btnk,
                  free_first_nlayers=free_first_nlayers,
                  free_last_nlayers=free_last_nlayers
                   )
    #compile the model
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    
    
    if use_btnk:
        #get the bottleneck features
        btnk_train,btnk_valid,btnk_test=get_bottleneck_features(btnk_path,arch=arch)
        
        checkpointer = ModelCheckpoint(filepath=new_model_path, 
                               verbose=2, save_best_only=True)
        model.fit(btnk_train, train_targets, 
              validation_data=(btnk_valid, valid_targets),
              epochs=epochs, 
              batch_size=20,
              callbacks=[checkpointer],
              verbose=2)
        #load the best weight
        model.load_weights(new_model_path)
        
        #get accuracy on test data
        # get index of predicted dog breed for each image in test set
        print('\n')
        predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in btnk_test]
        # report test accuracy
        test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
    else:
        #train use the full network
        #pre-process the data for Keras
        train_tensors = paths_to_tensor(train_files).astype('float32')/255
        valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
        test_tensors = paths_to_tensor(test_files).astype('float32')/255
        
        checkpointer = ModelCheckpoint(filepath=new_model_path, 
                             verbose=2, 
                             save_best_only=True)
        
        if augmentation:
            #data augmentation
            print("use augmentation")
            datagen_train = ImageDataGenerator(width_shift_range=0.2,#random shift horizaontally 10%
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True
                                  )
            #do nothing for valid
            datagen_valid = ImageDataGenerator()
            datagen_train.fit(train_tensors)
            datagen_valid.fit(valid_tensors)
            
            #fit model
            model.fit_generator(datagen_train.flow(train_tensors,train_targets,batch_size=20),
                         steps_per_epoch=train_tensors.shape[0]//20,
                         epochs=epochs,verbose=2,callbacks=[checkpointer],
                         validation_data=datagen_valid.flow(valid_tensors,valid_targets,batch_size=20),
                         validation_steps=valid_tensors.shape[0]//20)
        else:
            model.fit(train_tensors,
                   train_targets, 
                   validation_data=(valid_tensors, valid_targets),
                   epochs=epochs,
                   batch_size=20,
                   callbacks=[checkpointer],
                   verbose=2)
        
        #load the best model
        model.load_weights(new_model_path)
        # get index of predicted dog breed for each image in test set
        print('\n')
        #print("test_tensor shape ",test_tensors.shape)
        #print("test target shape ",test_targets.shape)
        predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
        # report test accuracy
        test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
        


      
  
    
     
    
     
    
    
    
    
    
            
        
    
    
    
    

   
        

