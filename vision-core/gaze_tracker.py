import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

GAZE_HISTORY_LENGTH = 5
SENSITIVITY = 250
RIGHT_EDGE_THRESHOLD = 0.8
LEFT_EDGE_THRESHOLD = 0.2
ANALYSIS_INTERVAL = 5
DETECTION_THRESHOLD = 0.5

gaze_history = deque(maxlen=GAZE_HISTORY_LENGTH)
right_edge_look_history = []
left_edge_look_history = []
last_analysis_time = time.time()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    h, w = image.shape[:2]
    
    cv2.rectangle(image, (int(LEFT_EDGE_THRESHOLD * w), 0), 
                 (int(RIGHT_EDGE_THRESHOLD * w), h), (100, 100, 100), 2)
    cv2.rectangle(image, (int(RIGHT_EDGE_THRESHOLD * w), 0), 
                 (w, h), (100, 100, 100), 2)
    
    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        
        left_eye_points = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_EYE])
        right_eye_points = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_EYE])
        left_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in LEFT_IRIS])
        right_iris = np.array([(mesh_points[p].x, mesh_points[p].y) for p in RIGHT_IRIS])
        left_center = left_iris.mean(axis=0)
        right_center = right_iris.mean(axis=0)
        
        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            return (A + B) / (2.0 * C)
        
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        
        if left_ear > 0.2 and right_ear > 0.2:
            def get_horizontal_gaze(eye_points, iris_center):
                eye_center = eye_points.mean(axis=0)
                gaze_x = (iris_center[0] - eye_center[0]) * SENSITIVITY
                screen_x = 0.5 + gaze_x
                return np.clip(screen_x, 0, 1)
            
            left_gaze_x = get_horizontal_gaze(left_eye_points, left_center)
            right_gaze_x = get_horizontal_gaze(right_eye_points, right_center)
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            
            gaze_history.append(gaze_x)
            smooth_gaze_x = np.mean(gaze_history)
            
            is_looking_right = smooth_gaze_x > RIGHT_EDGE_THRESHOLD
            right_edge_look_history.append(is_looking_right)
            is_looking_left = smooth_gaze_x < LEFT_EDGE_THRESHOLD
            left_edge_look_history.append(is_looking_left)
            
            gaze_pixel = (int(smooth_gaze_x * w), h // 2)
            color = (0, 255, 0) if is_looking_right else (0, 0, 255)
            cv2.circle(image, gaze_pixel, 10, color, -1)
            cv2.putText(image, f"Horizontal Gaze: {smooth_gaze_x:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_time = time.time()
            if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                right_look_ratio = sum(right_edge_look_history) / len(right_edge_look_history)
                if right_look_ratio >= DETECTION_THRESHOLD:
                    print(f"User looked at right edge {right_look_ratio*100:.1f}% of the time (last {ANALYSIS_INTERVAL} seconds)")
                
                left_look_ratio = sum(left_edge_look_history) / len(left_edge_look_history)
                if left_look_ratio >= DETECTION_THRESHOLD:
                    print(f"User looked at left edge {left_look_ratio*100:.1f}% of the time (last {ANALYSIS_INTERVAL} seconds)")
                
                right_edge_look_history = []
                left_edge_look_history = []
                last_analysis_time = current_time
    
    cv2.imshow('Horizontal Gaze Tracking', image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
