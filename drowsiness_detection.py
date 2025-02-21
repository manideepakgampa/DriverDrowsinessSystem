import os
import cv2
import dlib
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Initialize pygame for alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def get_mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def is_dark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < 50  # Adjust threshold as needed

# Capture video from webcam
cap = cv2.VideoCapture(0)
DROWSINESS_THRESHOLD = 0.25
MOUTH_THRESHOLD = 0.6
CLOSED_EYE_FRAMES = 20
YAWN_FRAMES = 15
frame_count = 0
yawn_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Check if the environment is dark and switch to IR mode
    if is_dark(frame):
        cv2.putText(frame, "IR Mode Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]
        
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = get_mouth_aspect_ratio(mouth)

        # Draw marker on face
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw square around face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if ear < DROWSINESS_THRESHOLD or mar > MOUTH_THRESHOLD:
            frame_count += 1
            yawn_count += 1 if mar > MOUTH_THRESHOLD else 0
            if frame_count >= CLOSED_EYE_FRAMES or yawn_count >= YAWN_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pygame.mixer.music.play()
        else:
            frame_count = 0
            yawn_count = 0
            cv2.putText(frame, "Driver Active - Safe to Drive", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
os._exit(0)
