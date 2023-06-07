import face_recognition
import os, sys
import cv2
import numpy as np
 

from facesJson  import *
from facesImage import *

from graphics import *
from recog    import *

from config import *


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    

    # recognition
    
    def __init__(self):
        self.reload_faces()
    
    def reload_faces(self):
        if LOAD_OPTION == 'IMAGE':
            self.encode_faces()
        elif LOAD_OPTION == 'JSON':
            self.apply_encodings_json(json_read_encodings())
        else:
            print("No import done")

    def reencode_faces(self):
        self.clear_encodings()
        self.reload_faces()

    
    def encode_faces(self):
        for image in os.listdir(FACE_PATH):
            face_encoding = image_read_face(image)
            if not isinstance(face_encoding, bool):
                print(f"Adding {image}")
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)


    def apply_encodings_json(self, encodings):
        for contents in encodings:
            print(contents)
            print("name", contents["name"])
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
    
    # interaction

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
                    save_image(frame, name)
                    self.update_faces(f'{name}{IMAGE_FORMAT}')
                    break
        if cv2.waitKey(1) == ord(KEY_RELOAD):
            print("Reloading face database...")
            standby_screen(frame)
            self.reencode_faces()
        if cv2.waitKey(1) == ord(KEY_EXPORT):
            print("Exporting to JSONs...")
            json_export_encodings(self.known_face_names, self.known_face_encodings)
        if cv2.waitKey(1) == ord(KEY_IMPORT):
            print("Importing from JSONs...")
            self.apply_encodings_json(json_read_encodings())
        if cv2.waitKey(1) == ord(KEY_CLEAR):
            print("Clearing encodings...")
            self.clear_encodings()
        return True






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

            display_faces(frame, self.face_locations, self.face_names)

        video_capture.release()
        cv2.destroyAllWindows()



    def process_stream(self, frame):
        if self.process_current_frame:
            modified = preprocess_frame(frame)
            loc_enc = search_faces(modified)

            self.face_locations = loc_enc["locations"]
            self.face_encodings = loc_enc["encodings"]
            self.face_names = try_recognize(self.face_encodings, self.known_face_encodings, self.known_face_names)

        self.process_current_frame = not self.process_current_frame
        