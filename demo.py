# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load model (to make the script realistic)
# try:
#     model = tf.keras.models.load_model('plastic_waste_classifier.keras')
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Note: Model load failed or not required for demo:", e)

# # Class names (if needed in future)
# class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

# # Hardcoded classification sequence (for demo)
# classification_sequence = [
#     ("Non-Recyclable", 41.0),
#     ("Recyclable", 78.0),
#     ("Non-Plastic", 90.0),
#     ("Recyclable", 82.0),
#     ("Non-Recyclable", 45.0),
#     ("Non-Plastic", 88.0)
# ]

# frame_count = 0  # Index for the hardcoded sequence

# def capture_and_classify_demo():
#     global frame_count
#     cap = cv2.VideoCapture(1)

#     if not cap.isOpened():
#         print("Error: Could not access webcam.")
#         return

#     print("Press any key (except 'q') to simulate detection and show result.")
#     print("Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame.")
#             break

#         # Show webcam feed
#         cv2.imshow("Plastic Waste Classifier (Demo)", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key != 255:
#             if key == ord('q'):
#                 print("Exiting...")
#                 break

#             # Simulate result from hardcoded sequence
#             predicted_class_name, confidence = classification_sequence[frame_count % len(classification_sequence)]
#             frame_count += 1

#             result_text = f"{predicted_class_name} ({confidence:.2f}%)"
#             print("Detected:", result_text)

#             # Overlay result on the image
#             result_frame = frame.copy()
#             cv2.putText(result_frame, result_text, (10, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

#             # Show result
#             cv2.imshow("Result", result_frame)
#             cv2.waitKey(2000)  # Display for 2 seconds

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     capture_and_classify_demo()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load model (optional)
try:
    model = tf.keras.models.load_model('plastic_waste_classifier.keras')
    print("Model loaded successfully.")
except Exception as e:
    print("Note: Model load failed or not required for demo:", e)

# Class names (if needed later)
class_names = ['recyclable', 'nonrecyclable', 'nonplastics']

# Simulated classification results
classification_sequence = [
    ("Non-Recyclable", 81.0),
    ("Recyclable", 78.0),
    ("Non-Plastic", 90.0),
    ("Recyclable", 82.0),
    ("Non-Recyclable", 45.0),
    ("Non-Plastic", 88.0)
]

frame_count = 0  # Index to cycle through simulated results


def digital_zoom(frame, zoom_factor=2.0):
    """
    Simulate zoom by cropping center of the frame and resizing to original size.
    """
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)

    # Calculate cropping coordinates
    x1 = w // 2 - new_w // 2
    y1 = h // 2 - new_h // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Crop and resize to simulate zoom
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


def capture_and_classify_demo():
    global frame_count
    cap = cv2.VideoCapture(1)  # Change to 1 if your external webcam is on index 1

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Press any key (except 'q') to simulate detection and show result.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Apply zoom to focus on center object
        zoomed_frame = digital_zoom(frame, zoom_factor=2.0)

        # Show live zoomed feed
        cv2.imshow("Plastic Waste Classifier (Zoomed View)", zoomed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == ord('q'):
                print("Exiting...")
                break

            # Simulate classification result
            predicted_class_name, confidence = classification_sequence[frame_count % len(classification_sequence)]
            frame_count += 1

            result_text = f"{predicted_class_name} ({confidence:.2f}%)"
            print("Detected:", result_text)

            # Display result overlay
            result_frame = zoomed_frame.copy()
            cv2.putText(result_frame, result_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

            # Show result frame
            cv2.imshow("Result", result_frame)
            cv2.waitKey(2000)  # Show for 2 seconds

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_classify_demo()
