from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

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


        

