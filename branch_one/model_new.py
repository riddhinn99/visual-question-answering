import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import keras
import os, argparse
import cv2, spacy, numpy as np
from keras.layers import Input, LSTM, Embedding, Dense
from keras.applications.vgg16 import VGG16
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras import backend as K
from keras.utils.vis_utils import plot_model
K.set_image_data_format('channels_first')
import csv

from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.utils import plot_model

import spacy.cli

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
class NewModel:
  def predict_in_class(self, incoming_image_path, incoming_question):
    with open('model_architecture_after_training_jupyter_car.json', 'r') as f:
        vqa_model = model_from_json(f.read())

    # Load weights into the new model
    vqa_model.load_weights('model_weights_after_training_jupyter_car.h5')

    #img = cv2.imread('sample_data_new/33.jpg')
    #cv2.imshow('image', img)
    #cv2.waitKey(0)

    #img = mpimg.imread('sample_data_new/33.jpg')
    #plt.imshow(img)

    def get_question_features_without_fd(question,word_embeddings):
        ''' For a given question, a unicode string, returns the time series vector
        with each word (token) transformed into a 300 dimension representation
        calculated using Glove Vector '''
        tokens = word_embeddings(question)
        question_tensor = np.zeros((30, 300))
        for j in range(len(tokens)):
            question_tensor[j,:] = tokens[j].vector
        return question_tensor

    def get_VQA_model():
        ''' Given the VQA model and its weights, compiles and returns the model '''

        # thanks the keras function for loading a model from JSON, this becomes
        # very easy to understand and work. Alternative would be to load model
        # from binary like cPickle but then model would be obfuscated to users
        with open('our_model_final.json','r') as f:
          vqa_model = keras.models.model_from_json(f.read())
        # vqa_model.load_weights(VQA_weights_file_name)
        vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return vqa_model

    def get_image_features_without_fd(image_file_name,model):
        ''' Runs the given image_file to VGG 16 model and returns the 
        weights (filters) as a 1, 4096 dimension vector '''
        img = image.load_img(image_file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
        fc2_features = model_extractfeatures.predict(x)
        image_features = fc2_features.reshape(4096)
        return image_features

    def get_image_model_without():
        ''' Takes the CNN weights file, and returns the VGG model update 
        with the weights. Requires the file VGG.py inside models/CNN '''
        image_model = VGG16(weights='imagenet', include_top=False)
        image_model.layers.pop()
        image_model.layers.pop()
        # this is standard VGG 16 without the last two layers
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # one may experiment with "adam" optimizer, but the loss function for
        # this kind of task is pretty standard
        image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
        return image_model

    def get_image_features_without_fd(image_file_name,model):
        ''' Runs the given image_file to VGG 16 model and returns the 
        weights (filters) as a 1, 4096 dimension vector '''
        img = image.load_img(image_file_name, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
        fc2_features = model_extractfeatures.predict(x)
        image_features = fc2_features.reshape(4096)
        return image_features


    def create_question_feature_test(text):
      question_feature=get_question_features_without_fd(text,word_embeddings)
      question_feature_processed = np.array(question_feature)  
      question_feature_processed = question_feature_processed.reshape((1,question_feature_processed.shape[0], question_feature_processed.shape[1]))
      return question_feature_processed

    def create_image_feature_test(image_path,model):
      image_features = get_image_features_without_fd(image_path,model)
      image_feature_processed = np.array(image_features)
      image_feature_processed = image_feature_processed.reshape((1,image_feature_processed.shape[0]))
      return image_feature_processed

    #spacy.cli.download("en_core_web_md")
    word_embeddings = spacy.load('en_core_web_md')
    model = vgg16.VGG16(weights='imagenet', include_top=True)

    training_datas = []
    with open('sample_data_new/training.csv') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        #print (row)
        training_datas.append(row)
    #training_datas = training_datas[1:]
    #len(training_datas)

    trainY= []
    for training_data in training_datas:
      img_id,text,output = training_data
      trainY.append(output)
    #print(len(trainY))
    #print(trainY)

    set_trainy = list(set(trainY))

    new_question = create_question_feature_test(incoming_question)
    new_image = create_image_feature_test(incoming_image_path,model)
    result=set_trainy[vqa_model.predict_classes([new_question,new_image])[0]]
    return result


