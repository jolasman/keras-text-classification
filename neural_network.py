import os, sys, pickle, inspect, itertools
import matplotlib.pyplot as plt, numpy as np, pandas as pd 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing import text, sequence
from keras import utils 
from utils import removeFile, countingWords

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def training():
    # One Epoch is when an Entire dataset is passed forward and backward through the neural network only ONCE!
    epochs = 1000
    #Total number of training example present in a sigle batch
    batch_size = 32

    training_percentage = 0.8

    # file path
    file_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    savings = file_path + '\\savings\\'
    dataset = file_path + '\\dataset\\'
    tokenizer_file = os.path.join(savings, 't.pickle')
    encoder_file = os.path.join(savings, 'e.pickle')
    class_file = os.path.join(savings, 'c.pickle')
    model_file = os.path.join(savings, 'm.h5')
    dataset_file = os.path.join(dataset, 'Youtube01-Psy.csv')

    #Create savings folder if not exists
    os.makedirs(os.path.dirname(tokenizer_file), exist_ok=True)

    removeFile(tokenizer_file)
    removeFile(encoder_file)
    removeFile(class_file)
    removeFile(model_file)


    data = pd.read_csv(dataset_file)

    print(data.head())

    training_size = int(len(data) * training_percentage)

    train_content = data['CONTENT'][:training_size]
    train_class = data['CLASS'][:training_size]

    test_content = data['CONTENT'][training_size:]
    test_class = data['CLASS'][training_size:]

    number_words_dataset = countingWords(train_content)

    tokenize = text.Tokenizer(num_words=number_words_dataset , char_level=False)

    tokenize.fit_on_texts(train_content)

    # tf-idf
    x_train = tokenize.texts_to_matrix(train_content, mode='tfidf')
    x_test = tokenize.texts_to_matrix(test_content, mode='tfidf')

    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenize,handle, protocol=pickle.HIGHEST_PROTOCOL)

    encoder = LabelEncoder()
    encoder.fit(train_class)

    y_train = encoder.transform(train_class)
    y_test = encoder.transform(test_class)

    with open(encoder_file, 'wb') as handle:
        pickle.dump(encoder,handle, protocol=pickle.HIGHEST_PROTOCOL)

    num_classes = np.max(y_train + 1)

    with open(class_file, 'wb') as handle:
        pickle.dump(num_classes,handle)

    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

#############################################################################################
    model = Sequential()
    model.add(Dense(num_classes * 8, input_shape=(number_words_dataset,), activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes * 4, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes * 2, activation = 'relu'))
    model.add(Dropout(0.2))

    #output layer
    model.add(Dense(num_classes, activation = 'softmax'))
#############################################################################################

    model.compile(loss= 'binary_crossentropy',
                    optimizer= 'adam',
                    metrics=['categorical_accuracy'])

    # model.compile(loss= 'categorical_crossentropy',
    #                 optimizer= 'adam',
    #                 metrics=['accuracy'])

    stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta= 0,
                                            patience=2,
                                            verbose=1,
                                            mode='auto',
                                            baseline=None)


    model_history = model.fit(x_train,y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_split=0.1,
                                callbacks=[stopper])

    score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=1)
    print("\n\n Test score: ", score[0])
    print("\n\n Test accuracy: ", score[1])

    model.save(model_file)


    # plot with losses
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.show()

    text_labels = encoder.classes_ 
    y_softmax = model.predict(x_test) 
    y_test_1d = []
    y_pred_1d = []

    for i in range(len(y_test)):
        probs = y_test[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)


    def plot_confusion_matrix(cm, classes,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=20)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)


    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(44,37))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.show()




if __name__ == '__main__':
    training()
