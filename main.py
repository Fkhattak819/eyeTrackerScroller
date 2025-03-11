
from eyeMovementDataCollectorClass import EyeMovementDataCollector
from eyeMovementModelClass import EyeMovementModel
from faceMarkers import FaceMeshDetector
import time
import cv2
import numpy as np
import os
import pickle


# Example usage for data collection and model training
def main():
    # 1. Collect training data
    collector = EyeMovementDataCollector()
    print("Starting data collection.")
    collector.collect_data()
    
    # 2. Train model on collected data
    print("\nTraining model on collected data...")
    model = EyeMovementModel()
    model.train(epochs=30)
    
    # 3. Save the trained model
    model.save_model()


    print("\nModel training complete and saved!")
    print("You can now use this model for real-time eye movement classification.\n")

    print("Eye tracker now running, Press 'q' to quit")

    live_prediction(collector,model)

# Live prediction function with face mesh drawing
def live_prediction(collector, model, sequence_length=10):
    # Initialize your eye feature collector and the face mesh detector
    face_detector = FaceMeshDetector(refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    sequence_data = []

    print("Starting live prediction with face landmarks. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:

            break

        # Get face landmarks and draw them on the frame
        frame, landmarks = face_detector.findMeshInFace(frame)
        # Draw left and right eye landmarks as green circles
        if "left_eye_landmarks" in landmarks:
            for lm in landmarks["left_eye_landmarks"]:
                cv2.circle(frame, lm, 3, (0, 255, 0), -1)
        if "right_eye_landmarks" in landmarks:
            for lm in landmarks["right_eye_landmarks"]:
                cv2.circle(frame, lm, 3, (0, 255, 0), -1)
        # Draw iris centers as blue circles
        if "left_iris_landmarks" in landmarks and len(landmarks["left_iris_landmarks"]) > 0:
            left_iris = np.array(landmarks["left_iris_landmarks"])
            left_pupil = left_iris.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(left_pupil), 4, (255, 0, 0), -1)
        if "right_iris_landmarks" in landmarks and len(landmarks["right_iris_landmarks"]) > 0:
            right_iris = np.array(landmarks["right_iris_landmarks"])
            right_pupil = right_iris.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(right_pupil), 4, (255, 0, 0), -1)
        
        # Extract features using existing EyeMovementDataCollector
        features = collector.extract_features(frame)
        if features is not None:
            sequence_data.append(features)

        # Overlay sequence progress on the frame
        cv2.putText(frame, f"Sequence: {len(sequence_data)}/{sequence_length}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # When the sequence is full, predict and reset the sequence
        if len(sequence_data) >= sequence_length:
            prediction = model.predict_sequence(sequence_data)
            print("Prediction:", prediction)
            cv2.putText(frame, f"Predicted: {prediction['position']}, Confidence: {prediction['confidence']}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            sequence_data = []  # Reset sequence data for the next prediction

        cv2.imshow("Live Prediction with Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()