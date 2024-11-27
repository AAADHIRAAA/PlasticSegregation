import numpy as np
import cv2
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('plastic_waste_classifier.keras')
print("Model loaded successfully.")

# Define class names
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

# Function to preprocess frames for model input
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model's input size
    frame_array = img_to_array(frame_resized)  # Convert to array
    frame_scaled = frame_array / 255.0  # Normalize pixel values
    frame_expanded = np.expand_dims(frame_scaled, axis=0)  # Add batch dimension
    return frame_expanded

# Function to predict the waste category
def predict_waste_category(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

# Use laptop's inbuilt camera for testing
def test_with_camera():
    cap = cv2.VideoCapture(1)  # Open webcam (0 is the default camera)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Predict waste category
        predicted_class_name, confidence = predict_waste_category(frame)

        # Display predictions on the frame
        text = f"{predicted_class_name} ({confidence:.2f}%)"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Plastic Waste Detection', frame)  # Show the frame

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Use a sample image file for testing
def test_with_image(image_path):
    image = cv2.imread(image_path)
    predicted_class_name, confidence = predict_waste_category(image)
    print(f"Predicted Class: {predicted_class_name} ({confidence:.2f}%)")

    # Display the image with the prediction
    text = f"{predicted_class_name} ({confidence:.2f}%)"
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Plastic Waste Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("Choose mode:")
    print("1. Test with Camera")
    print("2. Test with Image File")

    mode = input("Enter your choice (1 or 2): ")

    if mode == '1':
        test_with_camera()
    elif mode == '2':
        image_path = input("Enter the path to the image file: ")
        test_with_image(image_path)
    else:
        print("Invalid choice. Exiting...")
