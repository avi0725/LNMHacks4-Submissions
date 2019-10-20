import keras
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical 
from keras.optimizers import Adam
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report

classifier=Sequential()

classifier.add(Conv2D(16,(3,3),input_shape=(64,64,3),activation='relu',padding='same'))
classifier.add(Conv2D(16,(3,3),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Conv2D(32,(3,3),activation='relu',padding='same'))
classifier.add(Conv2D(32,(3,3),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Conv2D(64,(3,3),activation='relu',padding='same'))
classifier.add(Conv2D(64,(3,3),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Conv2D(96,(3,3),activation='relu',padding='same'))
classifier.add(Conv2D(96,(3,3),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Conv2D(128,(3,3),activation='relu',padding='same'))
classifier.add(Conv2D(128,(3,3),activation='relu',padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2),padding='same'))

classifier.add(Flatten())

classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('G:\X-RAY\chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
validation_set = test_datagen.flow_from_directory(
        'G:\X-RAY\chest_xray/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'G:\X-RAY\chest_xray/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


classifier.fit_generator(training_set, steps_per_epoch=5216/32, epochs=10, validation_data = validation_set, validation_steps=624/32)

    
test_loss,test_accuracy=classifier.evaluate_generator(test_set,steps=624)
print('The testing loss is :',test_loss*100, '%')
print('The testing accuracy is :',test_accuracy*100, '%')


import numpy as np
from keras.preprocessing import image
test_image=image.load_img('G:/X-RAY/chest_xray/test/person157_bacteria_735.jpeg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction="PNEUMONIA"
else:
    prediction="NORMAL"

#The testing loss is : 40.100532800002036 %
#The testing accuracy is : 88.29087921117502 %