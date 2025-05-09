import cv2
import mediapipe as mp
import numpy as np
import time

EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 2
CONSECUTIVE_BLINKS_THRESHOLD = 3
BLINK_WINDOW_TIME = 3.0

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

blink_counter = 0
frame_counter = 0
last_blink_time = time.time()
blink_sequence = 0

cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye_landmarks, landmarks):
    p1 = np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y])
    p2 = np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y])
    p3 = np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y])
    p4 = np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y])
    p5 = np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y])
    p6 = np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y])
    
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        left_ear = eye_aspect_ratio(LEFT_EYE_INDICES, landmarks)
        right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < EYE_AR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
                current_time = time.time()
                
                if current_time - last_blink_time <= BLINK_WINDOW_TIME:
                    blink_sequence += 1
                    if blink_sequence >= CONSECUTIVE_BLINKS_THRESHOLD:
                        print("Triple blink detected!")
                        blink_sequence = 0
                else:
                    blink_sequence = 1
                
                last_blink_time = current_time
            
            frame_counter = 0
        
        cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Blinks: {blink_counter}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Sequence: {blink_sequence}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if blink_sequence >= CONSECUTIVE_BLINKS_THRESHOLD:
            cv2.putText(image, "TRIPLE BLINK DETECTED!", (w//2 - 150, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imshow('Blink Detection', image)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

