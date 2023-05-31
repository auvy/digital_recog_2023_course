import face_recognition
import os, sys
import cv2
import numpy as np

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
        self.encode_faces()



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
        self.known_face_names = []
        self.known_face_encodings = []
        self.encode_faces()
        
    
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
                    # self.reencode_faces()
                    self.update_faces(f'{name}{IMAGE_FORMAT}')
                    break
        return True



    def run(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')
            
        print(f'Press {KEY_QUIT} to close app')
        print(f'Press {KEY_SAVE} if you want to screenshot')

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
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

            # Calculate the shortest distance to face
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