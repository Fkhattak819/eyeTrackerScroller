import mediapipe as mp 
import os
import cv2
import numpy as np
import time
import pickle

class EyeMovementDataCollector:
    def __init__(self, data_dir = "eye_movement_data"):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces = 1,
            refine_landmarks = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

        # Eye landmark indices
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

         # Data collection
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.eye_positions = ["neutral", "looking_up", "looking_down", "scroll"]

        for positions in self.eye_positions:
            os.makedirs(os.path.join(data_dir, positions), exist_ok=True)

        self.current_position = None
        self.recording = False
        self.sequence_data = []
        self.sequence_length = 10

    def extract_features(self,frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0].landmark

         # Extract features from both eyes
        features = []

        # Left eye features
        left_eye_points = [face_landmarks[i] for i in self.LEFT_EYE_INDICES]
        left_eye_center_x = np.mean([point.x for point in left_eye_points])
        left_eye_center_y = np.mean([point.y for point in left_eye_points])
        
        # Right eye features
        right_eye_points = [face_landmarks[i] for i in self.RIGHT_EYE_INDICES]
        right_eye_center_x = np.mean([point.x for point in right_eye_points])
        right_eye_center_y = np.mean([point.y for point in right_eye_points])

         # Eye openness (height)
        left_eye_top = face_landmarks[159].y
        left_eye_bottom = face_landmarks[145].y
        left_eye_height = left_eye_bottom - left_eye_top
        
        right_eye_top = face_landmarks[386].y
        right_eye_bottom = face_landmarks[374].y
        right_eye_height = right_eye_bottom - right_eye_top

        eye_distance = np.sqrt((right_eye_center_x - left_eye_center_x)**2 + 
                               (right_eye_center_y - left_eye_center_y)**2)
        
        # Compile features
        features = [
            left_eye_center_x, left_eye_center_y,
            right_eye_center_x, right_eye_center_y,
            left_eye_height, right_eye_height,
            eye_distance
        ]

         # Add positions of key eye landmarks for more detailed tracking
        for idx in [159, 145, 386, 374]:  # Top/bottom points of both eyes
            features.extend([face_landmarks[idx].x, face_landmarks[idx].y])
        
        return features
        
    def collect_data(self):
        cap = cv2.VideoCapture(0)

        print("\nEye Movement Data Collection")
        print("----------------------------")
        print("Press keys to control recording:")
        print("1-4: Select class (1=neutral, 2=looking_up, 3=looking_down, 4=scroll_gesture)")
        print("Space: Start/stop recording sequence")
        print("ESC: Exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            

            features = self.extract_features(frame)

            # Display info
            status = f"Class: {self.current_position if self.current_position else 'None'}"
            status += f" | Recording: {'YES' if self.recording else 'NO'}"
            status += f" | Frames: {len(self.sequence_data)}/{self.sequence_length}"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Record data if active
            if self.recording and features is not None:
                self.sequence_data.append(features)
                
                # Complete sequence
                if len(self.sequence_data) >= self.sequence_length:
                    self.save_sequence()
                    self.sequence_data = []
                    self.recording = False
            
            cv2.imshow('Eye Movement Data Collection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # Space
                if self.current_position:
                    self.recording = not self.recording
                    if self.recording:
                        print(f"Started recording sequence for class: {self.current_position}")
                        self.sequence_data = []
                    else:
                        print("Stopped recording")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                idx = key - ord('1')
                if idx < len(self.eye_positions):
                    self.current_position = self.eye_positions[idx]
                    print(f"Selected class: {self.current_position}")
                
            
        cap.release()
        cv2.destroyAllWindows()

    def save_sequence(self):
        """Save a complete sequence to file"""
        if not self.current_position or not self.sequence_data:
            return
        
        # Create filename with timestamp
        filename = f"{int(time.time())}.pkl"
        filepath = os.path.join(self.data_dir, self.current_position, filename)
        
        # Save sequence
        with open(filepath, 'wb') as f:
            pickle.dump(self.sequence_data, f)
            
        print(f"Saved sequence to {filepath}")










#---------------------------------------------------------------------------------------------

    def save_active_sequence(self):
        cap = cv2.VideoCapture(0)

        print("\nEye Movement Data Collection")
        print("----------------------------")
        print("Press keys to control recording:")
        print("1-4: Select class (1=neutral, 2=looking_up, 3=looking_down, 4=scroll_gesture)")
        print("Space: Start/stop recording sequence")
        print("ESC: Exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            

            features = self.extract_features(frame)

            # Display info
            status = f"Class: {self.current_position if self.current_position else 'None'}"
            status += f" | Recording: {'YES' if self.recording else 'NO'}"
            status += f" | Frames: {len(self.sequence_data)}/{self.sequence_length}"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Record data if active
            if self.recording and features is not None:
                self.sequence_data.append(features)
                
                # Complete sequence
                if len(self.sequence_data) >= self.sequence_length:
                    self.save_sequence()
                    self.sequence_data = []
                    self.recording = False
            
            cv2.imshow('Eye Movement Data Collection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # Space
                if self.current_position:
                    self.recording = not self.recording
                    if self.recording:
                        print(f"Started recording sequence for class: {self.current_position}")
                        self.sequence_data = []
                    else:
                        print("Stopped recording")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                idx = key - ord('1')
                if idx < len(self.eye_positions):
                    self.current_position = self.eye_positions[idx]
                    print(f"Selected class: {self.current_position}")
                
            
        cap.release()
        cv2.destroyAllWindows()


    def save_active__sequence(self):
        """Save a complete sequence to file"""
        if not self.current_position or not self.sequence_data:
            return
        
        # Create filename with timestamp
        filename = f"{int(time.time())}.pkl"
        filepath = os.path.join(self.data_dir, self.current_position, filename)
        
        # Save sequence
        with open(filepath, 'wb') as f:
            pickle.dump(self.sequence_data, f)
            
        print(f"Saved sequence to {filepath}")
            
