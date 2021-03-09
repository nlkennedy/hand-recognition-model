# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
# units = number of classes
classifier.add(Dense(units=4, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

batch_size = 5

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

training_set = train_datagen.flow_from_directory('data/train/',
                                                 target_size=(64, 64),
                                                 batch_size=batch_size,
                                                 color_mode='grayscale',
                                                 class_mode='categorical',
                                                 subset='training')

test_set = test_datagen.flow_from_directory('data/train/', # same as train not test
                                            target_size=(64, 64),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            subset='validation') 
classifier.fit(
        training_set,
        batch_size=5,
        steps_per_epoch=training_set.samples // batch_size,
        epochs=20,
        validation_data=test_set,
        validation_steps=test_set.samples // batch_size,
        verbose=2
)

classifier.save("model.h5")
