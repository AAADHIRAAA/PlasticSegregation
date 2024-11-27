import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Set paths for training and validation data
train_dir = 'samples/train'
validation_dir = 'samples/test'

# Define class names
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

# Image data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of data
batch_size = 16
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names
)

# Load pre-trained MobileNetV3 model
base_model = tf.keras.applications.MobileNetV3Large(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output_layer = Dense(len(class_names), activation='softmax')(x)

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=output_layer)

# Freeze pre-trained layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training the model...")
initial_epochs = 5
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=initial_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
print("Fine-tuning the model...")
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=total_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('plastic_waste_classifier.keras')

