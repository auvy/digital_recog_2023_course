import cv2

FACE_PATH = 'faces'
SAVE_PATH = FACE_PATH

KEY_QUIT = 'q'
KEY_SAVE = 'g'

# dividing on it can speed up recognition
SCALE_SIZE = 4

# forbidden symbols for windows filenames
FORBIDDEN_CHARACTERS = [" ", "<", ">", '"', "/", "\\", "|", "?", "*"]

# for cv2
BGR_RED   = (0,0,255)
BGR_WHITE = (255,255,255)
BGR_CYAN  = (255,255,0)
BGR_BLACK = (0,0,0)

CV2_DUPLEX = cv2.FONT_HERSHEY_DUPLEX


IMAGE_FORMAT = '.jpg'