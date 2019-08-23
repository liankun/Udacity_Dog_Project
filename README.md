[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/example1.png "example one"
[image3]: ./images/example2.png "example two"

### Project Overview
This is my final project for the Udacity Data Science Program.The main goal of this project is writing a algorithm to identify the resembling dog breed.The main procedure will be followed as dog_app.ipynb.The algorithm will be deployed as an application
![Sample Output][image1]



### Install Instruction
1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/liankun/Udacity_Dog_Project.git
cd Udacity_Dog_Project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/Udacity_Dog_Project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/Udacity_Dog_Project/lfw`. 

4. Download the following bottleneck features for the dog dataset. 
- [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz)
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

Place it in the repo, at location `path/to/Udacity_Dog_Project/bottleneck_features`.

5. The running environment is listed in env.txt

### File description
* dog_app.ipynb: jupyter notebook of this project which consists of the main procedures to follow
* env.txt : running environment
* saved_models/: the saved model after training
* papers/: some papers related to the face detection
* images/: example used in this project
* test_images/: test image examples
* haarcascades/haarcascade_frontalface_alt.xml: parameters used in the cv2 face detector
* extract_bottleneck_features.py: help function used in notebook
* app/: the application of this project
   - haarcascades/haarcascade_frontalface_alt.xml: parameters used in the cv2 face detector
   - Utility.py: help function for the application
   - extract_bottleneck_features.py: help function to extract bottleneck features for an image
   - predict_breed.py : main application for algorithm
   - config.json : the default setting of the application, this will changed after each settings
   - dog_names.pk : pickle files for a list of dog names
   - train_files(valid,test).pk, train(valid,test)_targets.pk,: numpy of images path and targets
### Application Usage
This application integrates prediction, training as well as architecture selection.The Application is based on transfer learning. There are four architectures: VGG19,ResNet50,Inception and Xception. The layers followed by these networks are Average Pooling, full connected hidden layer and full connected classification layer.
* Some Configurations:
   - `--input` : the input image for prediction
   - `--show_image`: this will show original image as well as most similar image.The way to do this is to loop over the training images, using the 
               hidden layer as the representation and get the highest similarity 
   - `--checkpoint` : the saving model directory.
   - `--training` : option for training
   - `--files_path`: the path for the dog images, for this is `path/to/Udacity_Dog_Project/dogImages`
   - `--bottleneck_feature_path` : the path for bottleneck features,`path/to/Udacity_Dog_Project/bottleneck_features`.
   - `--face_detector` : There are three types of face detector: cv2, hog and cnn. For hog and cnn you can refer to this [link](https://github.com/ageitgey/face_recognition). The cnn needs large GPU memory, it may crashes if lack of memory
   - `--arch` : choose one of the four achitectures
   - `--augmentation` : use augmentation for training. The augmentation is only used for **Full Network Training**, which means bottleneck features will                  not be used
   - `--full_network` : use bottleneck feature or full network to train 
   - `--hidden_layer_nodes` : the number of hidden layer nodes
   - `--free_first_layers` : number of layers be freed for the selected architecture. in order to use this to train, use_bottleneck_feature need to be                       set to False
   - `--free_last_layers` : similar to previous one, but free the last n layers
   - `--epochs` : number of epochs for training
   - `--best_model`: the name of the best model used for prediction. The name rules are `weights.best.{1}.{2}.{3}.{4}.{5}.hdf5`
               1. architecture
               2. number of hidden layer nodes
               3. number of first layers to be freed 
               4. number of last layers to be freed
               5. use bottleneck feature or full network
               **after each training, the best_model will set to be the new one**
   - `--create_pickle_files`: create pickle for train, test and valid. when changing the train files directory, should use this option
   - `--verbose` : print the configuration
* some examples: 
   - training use the Xception with hidden layer nodes 512, and use the bottleneck features:<br />
   `python predict_breed.py --training --hidden_layer_nodes=256 --arch='Xception'`
   - predict use the model from previous step, use face detector 'hog'. if you want to show image you can add `--show_image` opion:<br />
   ` python predict_breed.py --input='../test_images/test4.jpg' --face_detector='hog'`
   - if you want to show image, you can add `--show_image` option.It will take some time.<br />
   ` python predict_breed.py --input='../test_images/test4.jpg' --face_detector='cnn' --show_image` 
   - predict use another trained model: <br />
   `python predict_breed.py --input='../test_images/test4.jpg' --face_detector='hog' --best_model='weights.best.Xception.128.0.0.Bottleneck.hdf5'`
   - If you have a powerful GPU, you can train with augmentation and free serval layers: <br />
   `python predict_breed.py --verbose --training --hidden_layer_nodes=512 --full_network --arch='Inception' --free_first_layers=2 --free_last_layers=2  --augmentation --epochs=10`
* Here are two examples (by using Xception and number of hidden nodes=256), by enabling show image option
![example one][image2]
![example two][image3]

### Summary
As you see, the model seems to work well on the images show above. For the classification on dogs, we can get a accuracy of 83%. The model can get overtrained very easily. One solution is to get more data and besides embeding may be a good choice.  

### License
This project is licensed under the terms of the MIT license



