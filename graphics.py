import cv2
import math


from config import *
from facesJson import *

def increase_brightness(frame, value=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame


def display_faces(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale back
        top    *= SCALE_SIZE
        right  *= SCALE_SIZE
        bottom *= SCALE_SIZE
        left   *= SCALE_SIZE

        # Create the frame with the name
        cv2.rectangle(frame, (left, top),         (right, bottom), BGR_CYAN, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), BGR_CYAN, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), CV2_DUPLEX, 0.8, BGR_RED, 1)

    cv2.imshow('Face Recognition', frame)

def standby_screen(frame):
    height = len(frame) # height
    width = len(frame[0]) # width

    cv2.rectangle(frame, (0, 0), (width, height), BGR_CYAN, cv2.FILLED)
    cv2.putText(frame, "Please stand by...", (math.floor(width/3), math.floor(height/3)), CV2_DUPLEX, 0.8, BGR_RED, 1)
    cv2.imshow('Face Recognition', frame)