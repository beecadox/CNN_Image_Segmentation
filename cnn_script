# -------------------------
# needed imports
# ---------------------------
import keras
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as split
from helper_functions import load_images_from_folder, show_confusion_matrix

# -------------------------
# used functions
# ---------------------------


def int_to_categorical(a, b, c, num):
    a = keras.utils.to_categorical(a, num)
    b = keras.utils.to_categorical(b, num)
    c = keras.utils.to_categorical(c, num)
    return a, b, c


# function used for predicting labels with a cnn model based on images
def predict_scores(x_for_pred, cnn_model):
    y_predictions_vectorized = cnn_model.predict(x_for_pred)
    y_pred = np.argmax(y_predictions_vectorized, axis=1)
    return y_pred


# function that saves all the results in an excel
def scores_excel_file(train_y, y_pred_train, test_y, y_pred_test, ratio_per):
    title = "Results for ratio" + ratio_per

    labels = np.unique(train_y)
    show_confusion_matrix(train_y, y_pred_train, "CNN-Train-" + title, labels)
    show_confusion_matrix(test_y, y_pred_test, "CNN-Test-" + title, labels)
    # calculate the scores
    acc_train = accuracy_score(train_y, y_pred_train)
    acc_test = accuracy_score(test_y, y_pred_test)
    pre_train = precision_score(train_y, y_pred_train, average='macro')
    pre_test = precision_score(test_y, y_pred_test, average='macro')
    rec_train = recall_score(train_y, y_pred_train, average='macro')
    rec_test = recall_score(test_y, y_pred_test, average='macro')
    f1_train = f1_score(train_y, y_pred_train, average='macro')
    f1_test = f1_score(test_y, y_pred_test, average='macro')
    data_for_xls = [[ratio_per, str(acc_train), str(pre_train), str(rec_train), str(f1_train),
                     str(acc_test), str(pre_test), str(rec_test), str(f1_test)]]
    for data in data_for_xls:
        sheet.append(data)

    print('Accuracy scores of CNN model with ratio ' + ratio_per + ' are:',
          'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
    print('Precision scores of CNN model with ratio ' + ratio_per + ' are:',
          'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
    print('Recall scores of CNN model with ratio ' + ratio_per + ' are:',
          'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
    print('F1 scores of CNN model with ratio ' + ratio_per + ' are:',
          'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
    print('')


# function used for fitting the model
def fitting_cnn_model(input_image_size, batch_size, epochs, baseNumOfFilters, x_train, y_train,
                      x_test, y_test, x_val, y_val, model_name):

    inputs = keras.layers.Input((input_image_size[0], input_image_size[1], input_image_size[2]))

    s = keras.layers.Lambda(lambda x: x / 255)(inputs)  # normalize the input
    conv1 = keras.layers.Conv2D(filters=baseNumOfFilters, kernel_size=(13, 13))(s)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(filters=baseNumOfFilters * 2, kernel_size=(7, 7))(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = keras.layers.Conv2D(filters=baseNumOfFilters * 4, kernel_size=(3, 3))(pool2)
    drop3 = keras.layers.Dropout(0.25)(conv3)
    flat1 = keras.layers.Flatten()(drop3)
    dense1 = keras.layers.Dense(128, activation='relu')(flat1)
    outputs = keras.layers.Dense(y_train.shape[1], activation='softmax')(dense1)

    CNNmodel = keras.Model(inputs=[inputs], outputs=[outputs])
    CNNmodel.compile(optimizer='sgd',
                     loss=keras.losses.categorical_crossentropy,
                     metrics=['accuracy'])
    # print model summary
    CNNmodel.summary()

    # fit model parameters, given a set of training data
    callbacksOptions = [
        keras.callbacks.EarlyStopping(patience=15, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
        keras.callbacks.ModelCheckpoint('OutputFiles/Models/tmpCNN.h5', verbose=1, save_best_only=True, save_weights_only=True)]

    CNNmodel.fit(x_train, y_train,
                 batch_size=batch_size,
                 shuffle=True, epochs=epochs, verbose=1,
                 callbacks=callbacksOptions,
                 validation_data=(x_val, y_val))

    # calculate some common performance scores
    score = CNNmodel.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # saving the trained model
    CNNmodel.save(model_name)


# Excel file info - headers - name of the sheet etc
headers = ['Train Data ratio', 'Accuracy(tr)', ' Precision(tr)', 'Recall(tr)', 'F1 score(tr)',
           'Accuracy(te)', ' Precision(te)', 'Recall(te)', 'F1 score(te)']
workbook_name = 'OutputFiles/Results/ccn_classification_scores.xls'
wb = Workbook()
sheet = wb.active
sheet.title = 'Classification Scores'
sheet.append(headers)
row = sheet.row_dimensions[1]
row.font = Font(bold=True)

input_folder = 'M:/Datasets/Monkeys/Images'
input_image_size = [100, 100, 3]  # define the FIXED size that CNN will have as input
test_size = [0.4, 0.2]  # size of test set : 20%, 40%
model_names = ['OutputFiles/Models/monkeyCNN60.h5', 'OutputFiles/Models/monkeyCNN80.h5']
lb = preprocessing.LabelBinarizer()

for i in range(len(test_size)):

    x, y = load_images_from_folder(input_folder, input_image_size)
    print("Dataset Loading done!")
    print("Dataset split")
    xx, x_test, yy, y_test = split(x, y, test_size=0.3, random_state=3)
    x_train, x_val, y_train, y_val = split(xx, yy, test_size=0.1, random_state=4)

    # define some CNN parameters
    batch_size = 100
    num_classes = np.unique(y_train).__len__()
    epochs = 30
    baseNumOfFilters = 128
    print("Batch Size " + str(batch_size))
    print("Number of epochs " + str(epochs))
    print("Base number of filters used in conv2D" + str(baseNumOfFilters))
    print("=======================================")
    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train, y_test, y_val = int_to_categorical(y_train, y_test, y_val, num_classes)

    fitting_cnn_model(input_image_size, batch_size, epochs, baseNumOfFilters, x_train, y_train,
                      x_test, y_test, x_val, y_val, model_names[i])

    # loading a trained model & use it over test data
    loaded_model = keras.models.load_model(model_names[i])
    train_per = str(100 - (test_size[i] * 100)) + "%"
    y_train_pred = predict_scores(x_train, loaded_model)
    y_test_pred = predict_scores(x_test, loaded_model)
    scores_excel_file(np.argmax(y_train, axis=1), y_train_pred, np.argmax(y_test, axis=1), y_test_pred, train_per)
wb.save(filename=workbook_name)

