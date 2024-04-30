import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))  #include_top =False: importing only the convolutional layers not the fully connected part
model.trainable = False
model= tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])
#print(model.summary())

def extract_features(img_path,model):
    img= image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)     #converting image to a numpy array
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)   #converts your input to the format suitable wrt imagenet dataset
    result = model.predict(preprocessed_img).flatten()  #result is a flattened one-dimensional array containing the raw feature vector extracted from the preprocessed image by the ResNet50 model.
    normalized_result = result /norm(result)     #normalize the values i.e., between 0 1o 1
    return normalized_result

#returns the total no. of images in the image folder and path of first five images

filenames=[]

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))
#print(len(filenames))
#print(filenames[0:5])
feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

#print(np.array(feature_list).shape)

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))