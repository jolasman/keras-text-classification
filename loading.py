import os
import sys
import pickle
import inspect

import numpy as np
import pandas as pd

from tensorflow import keras
from keras import utils
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from utils import loadFile, removeFile

def predicting():
    """Loads the trained model, tokenizer, encoder and number of classes files. Gives an console interface to input product description and shows the 4 best
    results for it"""
    file_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    savings = file_path + '\\savings\\' 
    tokenizer_file = os.path.join(savings,"t.pickle")
    encoder_file = os.path.join(savings,"e.pickle")
    model_file = os.path.join(savings,"m.h5")
    classes_file = os.path.join(savings,"c.pickle")
    number_words_showing = 50

    print("\nGetting model data...")
    tokenizer_new = loadFile(tokenizer_file)
    encoder_new = loadFile(encoder_file)
    num_classes = loadFile(classes_file)

    if os.path.exists(model_file):
        model = load_model(model_file)
        print("\nModel loaded successfully...")
    else:
        print("There are no enough files to start predicting; please train the model first.")
        sys.exit()

    print("\nData loaded...")

    print("\n******************************************************************************")
    print("\n***************************   AI Testing Program   ***************************")
    print("\n******************************************************************************")
    print("Enter exit to close the program...")

    text_labels = encoder_new.classes_
    while True:
        user_input = input("\n\nPlease enter the sentence to analyse: \n\n")
        if user_input == "exit":
            break
        else:
            print("\n\nEvaluating input: ", user_input[:number_words_showing], "...\n")
            dataframe_input_test = pd.Series([user_input])
            x_test = tokenizer_new.texts_to_matrix(dataframe_input_test, mode='tfidf')
            prediction = model.predict(np.array([x_test[0]]))
            sorting = (-prediction).argsort()
            sorted_ = sorting[0][:2]
            for value in sorted_:
                predicted_label = text_labels[value]
                prob = (prediction[0][value]) * 100
                prob = "%.2f" % round(prob,2)
                print("I have %s%% sure that it belongs to: %s" % (prob, predicted_label))




if __name__ == '__main__':
    predicting()
    