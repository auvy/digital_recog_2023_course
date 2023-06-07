import face_recognition
import os, sys
import cv2
import numpy as np
import math
 
from helper import *

from config import *
from facesJson import *
from graphics import *


def image_read_face(image):
    face_image = face_recognition.load_image_file(f"{FACE_PATH}/{image}")
    
    if not face_recognition.face_encodings(face_image):
        print(f"Failed to recognize {image}, skipping")
        return False
    
    face_encoding = face_recognition.face_encodings(face_image)[0]

    return face_encoding


def preprocess_frame(frame):
    # resize for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_SIZE, fy=1/SCALE_SIZE)

    # convert cv2 bgr to rgb
    rgb_small_frame = small_frame[:, :, ::-1]

    # increase brightness for better recognition
    brighter_frame = increase_brightness(rgb_small_frame, BRIGHTNESS)
    
    return brighter_frame


def search_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    return {"locations": face_locations, "encodings": face_encodings}


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def try_recognize(face_encodings, known_face_encodings, known_face_names):
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        confidence = '???'
        
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

        face_names.append(f'{name} ({confidence})')
    return face_names

    