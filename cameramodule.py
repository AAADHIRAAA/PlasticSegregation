import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, Activation 
from tensorflow.keras.models import Model 
import os

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

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compile the model again after unfreezing layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retrain the model with fine-tuning
history_fine_tuning = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

def preprocess_input(img):
    # Preprocess the image (resize, normalize, etc.)
    img = cv2.resize(img, (224, 224))  # Resize the image to match the input size of your model
    img = img.astype(np.float32) / 255.0  # Normalize pixel values between 0 and 1
    img = tf.keras.applications.mobilenet.preprocess_input(img)  # Preprocess input according to the MobileNet preprocessing
    return img


# Initialize the camera
cap = cv2.VideoCapture(0)

# Define class names
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

while True:
    ret, frame = cap.read()  # Read frame from camera
    
    # Preprocess the frame
    processed_frame = preprocess_input(frame)
    
    # Make prediction
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the captured image
    cv2.imshow('Captured Image', frame)
   
    
    # Print predicted class
    print("Predicted class:", predicted_class_name)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()