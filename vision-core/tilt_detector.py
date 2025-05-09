import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

FACE_TOP = 10
FACE_BOTTOM = 152
FACE_LEFT = 234
FACE_RIGHT = 454

TILT_ANGLE_THRESHOLD = 15
TILT_DURATION_THRESHOLD = 3
TILT_HISTORY_LENGTH = 15

tilt_history = deque(maxlen=TILT_HISTORY_LENGTH)
tilt_start_time = None
last_tilt_direction = None
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
    
    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        
        top = np.array([mesh_points[FACE_TOP].x * w, mesh_points[FACE_TOP].y * h])
        bottom = np.array([mesh_points[FACE_BOTTOM].x * w, mesh_points[FACE_BOTTOM].y * h])
        left = np.array([mesh_points[FACE_LEFT].x * w, mesh_points[FACE_LEFT].y * h])
        right = np.array([mesh_points[FACE_RIGHT].x * w, mesh_points[FACE_RIGHT].y * h])
        
        vertical_vector = bottom - top
        horizontal_vector = right - left
        
        tilt_angle = math.degrees(math.atan2(horizontal_vector[1], horizontal_vector[0]))
        
        if tilt_angle > 90:
            tilt_angle -= 180
        elif tilt_angle < -90:
            tilt_angle += 180
        
        tilt_history.append(tilt_angle)
        smoothed_tilt = np.mean(tilt_history)
        
        current_tilt_direction = None
        if smoothed_tilt > TILT_ANGLE_THRESHOLD:
            current_tilt_direction = "right"
        elif smoothed_tilt < -TILT_ANGLE_THRESHOLD:
            current_tilt_direction = "left"
        else:
            current_tilt_direction = "center"
        
        current_time = time.time()
        if current_tilt_direction == last_tilt_direction:
            if tilt_start_time is None:
                tilt_start_time = current_time
            elif tilt_start_time is not None and current_time - tilt_start_time >= TILT_DURATION_THRESHOLD:
                print(f"Head tilted {current_tilt_direction} for {TILT_DURATION_THRESHOLD} seconds!")
                tilt_start_time = None
        else:
            tilt_start_time = None
        
        last_tilt_direction = current_tilt_direction
        
        cv2.putText(image, f"Tilt: {smoothed_tilt:.1f}° {current_tilt_direction}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        tilt_indicator_x = w // 2 + int(smoothed_tilt * 2)
        cv2.line(image, (w // 2, h // 2), (tilt_indicator_x, h // 2), (0, 255, 255), 3)
    
    cv2.imshow('Head Tilt Tracking', image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

