import numpy as np
import cv2 # type: ignore
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, Activation 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import imagenet_utils 
from sklearn.metrics import confusion_matrix 
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt 


     

mobile = tf.keras.applications.mobilenet.MobileNet()

# def prepare_image(file):
#     img_path = 'samples/test/nonplastics/'
#     img = image.load_img(img_path + file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# Display the image
# img_path = 'samples/test/nonplastics/np6.jpeg'
# img = image.load_img(img_path, target_size=(224, 224))
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# Prepare and predict the image
# preprocessed_image = prepare_image('np6.jpeg')
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)

# print(results)
# Define directories for training and validation data
train_dir = 'samples/train'
validation_dir = 'samples/test'

# Image data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size
batch_size = 10

# Generate batches of training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['recyclable', 'nonrecyclable', 'nonplastics']
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['recyclable', 'nonrecyclable', 'nonplastics']
)

# Load pre-trained MobileNet model
base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False)

# Add custom classification head
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)  # Assuming 3 classes

# Combine base model and custom head
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# # Freeze pre-trained layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=5,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )
# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compile the model again after unfreezing layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),  # Adjust the learning rate if needed
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retrain the model with fine-tuning
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # You can adjust the number of epochs for fine-tuning
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Example of making predictions on new data
test_image_path = 'samples/test/recyclable/r16.jpg'
test_image = image.load_img(test_image_path, target_size=(224, 224))
test_image_array = image.img_to_array(test_image)
test_image_array_expanded_dims = np.expand_dims(test_image_array, axis=0)
preprocessed_test_image = tf.keras.applications.mobilenet.preprocess_input(test_image_array_expanded_dims)
prediction = model.predict(preprocessed_test_image)
print(prediction)
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']  # Define the class names

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(prediction)

# Get the class name corresponding to the predicted index
predicted_class_name = class_names[predicted_class_index]

print("Predicted class:", predicted_class_name)


