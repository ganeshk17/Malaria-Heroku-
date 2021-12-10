# Malaria-Heroku-

Model is prepared using CNN method which is deployed using Flask on Heroku Cloud platform

Inintially a model is prepared using CNN then if deployed using Flask and then finally deployed on Heroku platform


1 ) ----Model-----

# import libraries

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Function used for Classification of data
classifier = Sequential()

# Sizing the data in layers
classifier.add(Conv2D(32, (3, 3), input_shape = (60, 60, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier = Sequential()

# Final layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the data into single
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Used for Image processing
from keras.preprocessing.image import ImageDataGenerator

# data to splited into training and testing of data
train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   zoom_range = 0.3,
                                   horizontal_flip = True)
                                   
                                   
# Rescaling of data
test_data = ImageDataGenerator(rescale = 1./255)

train_set = train_data.flow_from_directory('cell_images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
                                                 
                                                 
  test_set = test_data.flow_from_directory('cell_images/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
                                            
# Trained data fited to test data with fit function
classifier.fit_generator(train_set,
                         steps_per_epoch = 5000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)
                         
                         
# model saved
from keras.models import load_model
classifier.save("malaria.h5")

# for image testing
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('cell_images/test.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Parasitized'
else:
    prediction = 'Uninfected'
    
2 ) ..........Flask.............
    
Once model is prepared then use Flask for model deployment.

3 ) .........Heroku.............

Finally connect model with Heroku
