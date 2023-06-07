import os, sys
import cv2

import json
 
from config import *


def json_export_encodings(known_face_names, known_face_encodings): #face_names, face_encodings
    for name, encoding in zip(known_face_names, known_face_encodings):
        encode_name = f'{name}.json'
        contents = {
            "name": name,
            "encoding": encoding.tolist()
        }
        json_object = json.dumps(contents, indent=4)
        with open(f'{ENCODE_PATH}/{encode_name}', 'w') as f:
            f.write(json_object)

def json_read_encodings():
    res = []
    for file in os.listdir(ENCODE_PATH):
        f = open(f'{ENCODE_PATH}/{file}', "r")
        contents = json.load(f)
        res.append(contents)
    return res