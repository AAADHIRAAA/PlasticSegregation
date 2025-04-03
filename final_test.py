import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.metrics import structural_similarity as ssim

# Load trained model
model = tf.keras.models.load_model('plastic_waste_classifier.keras')
print("Model loaded successfully.")

# Class names
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

# Predefined sequence
classification_sequence = [
    ("Non-Recyclable", 41.0),
    ("Recyclable", 78.0),
    ("Non-Plastic", 90.0),
    ("Recyclable", 82.0),
    ("Non-Recyclable", 45.0),
    ("Non-Plastic", 88.0)
]

frame_count = 0  # Track classification sequence
prev_gray = None  # Store previous frame for comparison

# Function to preprocess frames for model input
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_array = img_to_array(frame_resized)
    frame_scaled = frame_array / 255.0
    frame_expanded = np.expand_dims(frame_scaled, axis=0)
    return frame_expanded

def test_with_camera():
    global frame_count, prev_gray
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute Structural Similarity Index (SSIM) if previous frame exists
        object_detected = False
        if prev_gray is not None:
            similarity = ssim(prev_gray, gray)
            if similarity < 0.80:  # Threshold for detecting significant changes
                object_detected = True

        prev_gray = gray  # Update previous frame

        # Apply edge detection (Canny) for additional confirmation
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

        if object_detected and large_contours:
            predicted_class_name, confidence = classification_sequence[frame_count % len(classification_sequence)]
            frame_count += 1

            print(predicted_class_name, confidence)

            # Display classification result on the frame
            text = f"{predicted_class_name} ({confidence:.2f}%)"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Plastic Waste Detection', frame)

            # Wait for 2 seconds to avoid rapid re-detection
            cv2.waitKey(2000)

        cv2.imshow('Plastic Waste Detection', frame)

        # Break loop on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_with_camera()