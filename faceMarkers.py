import mediapipe as mp 
import cv2
import numpy as np

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, 
                 min_detection_con=0.5, min_tracking_con=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            self.refine_landmarks,
            self.min_detection_con,
            self.min_tracking_con
        )
        # Predefined Mediapipe indices for left and right eyes
        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]
        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]
        # Indices for iris landmarks (if refine_landmarks=True, these become available)
        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]

    def findMeshInFace(self, img):
        landmarks = {}
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                # Initialize lists for each set of landmarks
                landmarks["left_eye_landmarks"] = []
                landmarks["right_eye_landmarks"] = []
                landmarks["left_iris_landmarks"] = []
                landmarks["right_iris_landmarks"] = []
                landmarks["all_landmarks"] = []
                
                for i, lm in enumerate(faceLms.landmark):
                    h, w, _ = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks["all_landmarks"].append((x, y))
                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye_landmarks"].append((x, y))
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye_landmarks"].append((x, y))
                    if i in self.LEFT_IRIS_LANDMARKS:
                        landmarks["left_iris_landmarks"].append((x, y))
                    if i in self.RIGHT_IRIS_LANDMARKS:
                        landmarks["right_iris_landmarks"].append((x, y))
        return img, landmarks
    



