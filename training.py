import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Define classes
classes = ['Recyclable Plastic', 'Non-Recyclable Plastic', 'Non-Plastic']

def preprocess_frame(frame):
    """Resize and normalize the image."""
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to MobileNetV3 input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

# Open the laptop camera
cap = cv2.VideoCapture(0)  # 0 refers to the default camera

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Preprocess the captured frame
    processed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(processed_frame)
    class_index = np.argmax(predictions)  # Get the class with the highest probability
    class_label = classes[class_index]  # Get the corresponding label

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Plastic Waste Classification', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
