import argparse
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math
import asyncio
import websockets
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run PDF-Navigator with head-tilt, blink or gaze controls"
)
parser.add_argument(
    "--mode",
    choices=['head', 'blink', 'gaze'],
    required=True,
    help="Control mode (head, blink, or gaze)"
)
args = parser.parse_args()
current_mode = args.mode

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

FACE_TOP = 10
FACE_BOTTOM = 152
FACE_LEFT = 234
FACE_RIGHT = 454
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

TILT_ANGLE_THRESHOLD = 15
TILT_DURATION_THRESHOLD = 1
TILT_HISTORY_LENGTH = 15
tilt_history = deque(maxlen=TILT_HISTORY_LENGTH)
tilt_start_time = None
last_tilt_direction = None

EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 2
CONSECUTIVE_BLINKS_THRESHOLD = 3
BLINK_WINDOW_TIME = 1.0
blink_counter = 0
frame_counter = 0
last_blink_time = None
blink_timestamps = []

PAGE_FLIP_MESSAGE = ""
PAGE_FLIP_MESSAGE_TIME = 0
PAGE_FLIP_DISPLAY_DURATION = 2.0

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

async def send_command(command):
    logger.info(f"Attempting to send command: {command}")
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            await websocket.send(command)
            logger.info(f"Command '{command}' sent successfully")
    except Exception as e:
        logger.error(f"Error sending command: {e}")

def eye_aspect_ratio(eye_landmarks, landmarks):
    """Calculate the eye aspect ratio (EAR) for an eye"""
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

async def process_head_tilt(image, mesh_points, h, w):
    """Process head tilt detection"""
    global tilt_start_time, last_tilt_direction, PAGE_FLIP_MESSAGE, PAGE_FLIP_MESSAGE_TIME
    
    left = np.array([mesh_points[FACE_LEFT].x * w, mesh_points[FACE_LEFT].y * h])
    right = np.array([mesh_points[FACE_RIGHT].x * w, mesh_points[FACE_RIGHT].y * h])
    
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
            if current_tilt_direction == "right":
                PAGE_FLIP_MESSAGE = "Page flipped right"
                PAGE_FLIP_MESSAGE_TIME = current_time
                await send_command("next_page")
            elif current_tilt_direction == "left":
                PAGE_FLIP_MESSAGE = "Page flipped left"
                PAGE_FLIP_MESSAGE_TIME = current_time
                await send_command("prev_page")
            tilt_start_time = None
    else:
        tilt_start_time = None
    
    last_tilt_direction = current_tilt_direction
    
    cv2.putText(image, f"Tilt: {smoothed_tilt:.1f}° {current_tilt_direction}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    tilt_indicator_x = w // 2 + int(smoothed_tilt * 2)
    cv2.line(image, (w // 2, h // 2), (tilt_indicator_x, h // 2), (0, 255, 255), 3)
    
    current_time = time.time()
    if current_time - PAGE_FLIP_MESSAGE_TIME < PAGE_FLIP_DISPLAY_DURATION:
        cv2.putText(image, PAGE_FLIP_MESSAGE, (w//2 - 150, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return image

async def process_blink_detection(image, mesh_points, h, w):
    """Process blink detection"""
    global blink_counter, frame_counter, last_blink_time, blink_timestamps
    global PAGE_FLIP_MESSAGE, PAGE_FLIP_MESSAGE_TIME
    
    left_ear = eye_aspect_ratio(LEFT_EYE_INDICES, mesh_points)
    right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES, mesh_points)
    avg_ear = (left_ear + right_ear) / 2.0
    
    current_time = time.time()
    
    if avg_ear < EYE_AR_THRESHOLD:
        frame_counter += 1
    else:
        if frame_counter >= EYE_AR_CONSEC_FRAMES:
            blink_counter += 1
            blink_timestamps.append(current_time)
            
            blink_timestamps = [t for t in blink_timestamps if current_time - t <= BLINK_WINDOW_TIME]
            
            if len(blink_timestamps) >= CONSECUTIVE_BLINKS_THRESHOLD:
                PAGE_FLIP_MESSAGE = f"Page flipped ({len(blink_timestamps)} blinks)"
                PAGE_FLIP_MESSAGE_TIME = current_time
                await send_command("next_page")
                blink_timestamps = []
                blink_counter = 0
            
            last_blink_time = current_time
        
        frame_counter = 0
    
    cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Blinks: {len(blink_timestamps)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Window: {BLINK_WINDOW_TIME:.1f}s", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    current_time = time.time()
    if current_time - PAGE_FLIP_MESSAGE_TIME < PAGE_FLIP_DISPLAY_DURATION:
        cv2.putText(image, PAGE_FLIP_MESSAGE, (w//2 - 150, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return image

async def process_gaze_tracking(image, mesh_points, h, w):
    """Process gaze tracking"""
    global right_edge_look_history, left_edge_look_history, last_analysis_time
    global PAGE_FLIP_MESSAGE, PAGE_FLIP_MESSAGE_TIME
    
    cv2.rectangle(image, (int(RIGHT_EDGE_THRESHOLD * w), 0), 
                 (w, h), (100, 100, 100), 2)
    cv2.rectangle(image, (0, 0), 
                 (int(LEFT_EDGE_THRESHOLD * w), h), (100, 100, 100), 2)
    
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
                PAGE_FLIP_MESSAGE = "Page flipped right"
                PAGE_FLIP_MESSAGE_TIME = current_time
                await send_command("next_page")
                print(f"User looked at right edge {right_look_ratio*100:.1f}% of the time (last {ANALYSIS_INTERVAL} seconds)")
            
            left_look_ratio = sum(left_edge_look_history) / len(left_edge_look_history)
            if left_look_ratio >= DETECTION_THRESHOLD:
                PAGE_FLIP_MESSAGE = "Page flipped left"
                PAGE_FLIP_MESSAGE_TIME = current_time
                await send_command("prev_page")
                print(f"User looked at left edge {left_look_ratio*100:.1f}% of the time (last {ANALYSIS_INTERVAL} seconds)")
            
            right_edge_look_history = []
            left_edge_look_history = []
            last_analysis_time = current_time
    
    current_time = time.time()
    if current_time - PAGE_FLIP_MESSAGE_TIME < PAGE_FLIP_DISPLAY_DURATION:
        cv2.putText(image, PAGE_FLIP_MESSAGE, (w//2 - 150, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return image

async def video_processing(current_mode):
    logger.info(f"Starting video processing in {current_mode} mode")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open video capture")
            return
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logger.warning("Failed to capture frame")
                continue
            
            logger.debug("Processing frame...")
        
            image = cv2.flip(image, 1)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            h, w = image.shape[:2]
            
            cv2.putText(image, f"Mode: {current_mode}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if results.multi_face_landmarks:
                mesh_points = results.multi_face_landmarks[0].landmark
                
                if current_mode == 'head':
                    image = await process_head_tilt(image, mesh_points, h, w)
                elif current_mode == 'blink':
                    image = await process_blink_detection(image, mesh_points, h, w)
                elif current_mode == 'gaze':
                    image = await process_gaze_tracking(image, mesh_points, h, w)
            
            cv2.imshow('PDF Navigator', image)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
    except Exception as e:
        logger.error(f"Video processing error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video processing stopped")

async def main():
    logger.info(f"Starting PDF Navigator in {current_mode} mode")
    try:
        await video_processing(current_mode)
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())