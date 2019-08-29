# import libraries
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import model_from_json
from keras import applications
from keras import regularizers
from keras import optimizers
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

# set up parameters
img_width, img_height = 150, 150
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '>1000data/train'
validation_data_dir = '>1000data/validation'
test_data_dir = '>1000data/test'
nb_train_samples = 4800
nb_validation_samples = 600
nb_test_samples = 200
epochs = 100
batch_size = 10

########################### Bottleneck Features Extraction on VGG16 Function ##########################

# load Image Data Generator
datagen = ImageDataGenerator(
    rescale=1., featurewise_center=True)  # (rescale=1./255)
datagen.mean = np.array([103.939, 116.779, 123.68],
                        dtype=np.float32).reshape(1, 1, 3)

# build the VGG16 network
model = applications.VGG16(include_top=False, weights='imagenet')

# get bottleneck features on train set
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)

np.save('bottleneck_features_train.npy', bottleneck_features_train)

# get bottleneck features on validation set
generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // batch_size)

np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

# get bottleneck features on test set
generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_test = model.predict_generator(
    generator, nb_test_samples // batch_size)

np.save('bottleneck_features_test.npy', bottleneck_features_test)

print("Done with bottleneck features")

########################### Train Classifier Function ##########################

# load bottleneck features
train_data = np.load('bottleneck_features_train.npy')
train_labels = np.array(
    [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

validation_data = np.load('bottleneck_features_validation.npy')
validation_labels = np.array(
    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

# build new model
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

# savel model and weights
model_json = model.to_json()
with open("model_4800train_bottleNeck.json", "w") as json_file:
    json_file.write(model_json)
    print("saved model to disk")

model.save_weights(top_model_weights_path)


########################### Load Model Function ##########################

test_labels = np.array([0]*100 + [1]*100)
print("generator ready")

# load model and weights
json_file = open('model_4800train_bottleNeck.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
print("json read")

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
model.load_weights('bottleneck_fc_model.h5')
print("model ready")


########################### Plot Confusion Matrix Function ##########################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

########################### Predict Function ##########################


# predict on test data
score = model.predict_classes(bottleneck_features_test).reshape((1, 200))
print(score.size)
classname = score[0]

# show confusion matrix
cm = confusion_matrix(test_labels, score[0, :])
cm_plot_labels = ['healthy', 'sick']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

########################### Plot Image Function ##########################


def plots(ims, figsize=(9, 9), rows=1, interp=False, titles=None):

    f = plt.figure(figsize=figsize)

    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=12)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


########################### Plot Images with Prediction Function  ##########################

datagen2 = ImageDataGenerator(rescale=1. / 255)
test_batch = datagen2.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=200,
    shuffle=False)

test_imgs, test_labels = next(test_batch)
plots(test_imgs, figsize=(15, 100), rows=40, titles=classname)
