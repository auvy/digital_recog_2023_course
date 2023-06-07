import os, sys
import cv2

import json
 
from config import *
import face_recognition



def save_image(frame, name):
    cv2.imwrite(f'{SAVE_PATH}/{name}{IMAGE_FORMAT}', frame) 

