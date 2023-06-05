import face_recognition
import os, sys
import cv2
import numpy as np
import math

import json
 
from pprint import pprint

from helper import face_confidence
from config import *

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True



    def __init__(self):
        self.reload_faces()

    def reload_faces(self):
        if LOAD_OPTION == 'IMAGE':
            self.encode_faces()
        elif LOAD_OPTION == 'JSON':
            self.import_encodings()
        else:
            print("No import done")


    def encode_faces(self):
        for image in os.listdir(FACE_PATH):
            face_image = face_recognition.load_image_file(f"{FACE_PATH}/{image}")
            
            if not face_recognition.face_encodings(face_image):
                print(f"Failed to recognize {image}, skipping")
                continue
            
            face_encoding = face_recognition.face_encodings(face_image)[0]
            print(f"Adding {image}")

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)



    def reencode_faces(self):
        self.clear_encodings()
        self.reload_faces()
    
    
    def export_encodings(self):
        for name, encoding in zip(self.known_face_names, self.known_face_encodings):
            encode_name = f'{name}.json'
            contents = {
                "name": name,
                "encoding": encoding.tolist()
            }
            json_object = json.dumps(contents, indent=4)
            with open(f'{ENCODE_PATH}/{encode_name}', 'w') as f:
                f.write(json_object)
            
    def import_encodings(self):
        for file in os.listdir(ENCODE_PATH):
            f = open(f'{ENCODE_PATH}/{file}', "r")
            contents = json.load(f)
            print(contents["name"])
            print(np.array(contents["encoding"]))
            self.update_encodings(np.array(contents["encoding"]), contents["name"])
    
    def update_encodings(self, encoding, name):
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)

    def clear_encodings(self):
        self.known_face_names = []
        self.known_face_encodings = []
            

        

            
        
    
    def update_faces(self, image):
        face_image = face_recognition.load_image_file(f"{FACE_PATH}/{image}")
        
        if not face_recognition.face_encodings(face_image):
            print(f"Failed to recognize {image}, skipping")
            return
        
        face_encoding = face_recognition.face_encodings(face_image)[0]
        print(f"Adding {image}")

        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(image)



    def save_image(self, frame, name):
        cv2.imwrite(f'{SAVE_PATH}/{name}{IMAGE_FORMAT}', frame) 



    def cli(self, frame):
        if cv2.waitKey(1) == ord(KEY_QUIT):
            print("Quitting")
            return False
        if cv2.waitKey(1) == ord(KEY_SAVE):
            print(KEY_SAVE)
            name = FORBIDDEN_CHARACTERS[0]
            while True:
                if any(ext in FORBIDDEN_CHARACTERS for ext in name):
                    name = input("Enter the name of your face: ")
                    print(name)
                else:
                    self.save_image(frame, name)
                    self.update_faces(f'{name}{IMAGE_FORMAT}')
                    break
        if cv2.waitKey(1) == ord(KEY_RELOAD):
            print("Reloading face database...")
            self.standby_screen(frame)
            self.reencode_faces()
        if cv2.waitKey(1) == ord(KEY_EXPORT):
            print("Exporting to JSONs...")
            self.export_encodings()
        if cv2.waitKey(1) == ord(KEY_IMPORT):
            print("Importing from JSONs...")
            self.import_encodings()
        if cv2.waitKey(1) == ord(KEY_CLEAR):
            print("Clearing encodings...")
            self.clear_encodings()
        return True


    def standby_screen(self, frame):
        height = len(frame) # height
        width = len(frame[0]) # width

        cv2.rectangle(frame, (0, 0), (width, height), BGR_CYAN, cv2.FILLED)
        cv2.putText(frame, "Please stand by...", (math.floor(width/3), math.floor(height/3)), CV2_DUPLEX, 0.8, BGR_RED, 1)
        cv2.imshow('Face Recognition', frame)


    def run(self):
        video_capture = cv2.VideoCapture(CAMERA_INDEX)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')
            
        print(f'Press {KEY_QUIT} to close app')
        print(f'Press {KEY_SAVE} if you want to screenshot')
        print(f'Press {KEY_RELOAD} to update database')
        print(f'Press {KEY_EXPORT} to export faces as JSON')
        print(f'Press {KEY_IMPORT} to import JSONs')
        print(f'Press {KEY_CLEAR} to reset database')
    

        CLI = True
        while CLI:
            ret, frame = video_capture.read()
        
            CLI = self.cli(frame)

            self.process_stream(frame)

            self.display_faces(frame)

        video_capture.release()
        cv2.destroyAllWindows()



    def process_stream(self, frame):
        if self.process_current_frame:
            self.process_frame(frame)
        self.process_current_frame = not self.process_current_frame

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=1/SCALE_SIZE, fy=1/SCALE_SIZE)

        # convert cv2 bgr to rgb
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        self.try_recognize(self.face_names)
        
        
        
    def try_recognize(self, face_names):
        for face_encoding in self.face_encodings:
            name = "Unknown"
            confidence = '???'
            
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

            face_names.append(f'{name} ({confidence})')
    
        

    def display_faces(self, frame):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
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