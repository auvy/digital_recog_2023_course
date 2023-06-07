import cv2

FACE_PATH = 'faces'
SAVE_PATH = FACE_PATH

ENCODE_PATH = 'encodings'

KEY_QUIT   = 'q'
KEY_SAVE   = 'g'
KEY_RELOAD = 'r'
KEY_EXPORT = 'e'
KEY_IMPORT = 'i'
KEY_CLEAR  = 'c'

# recognition parameters
# downscaling by SCALE_SIZE helps speed up process
SCALE_SIZE = 4
# brightness, percentage, helps recognize
BRIGHTNESS = 20

# forbidden symbols for windows filenames
FORBIDDEN_CHARACTERS = [" ", "<", ">", '"', "/", "\\", "|", "?", "*"]

# for cv2
BGR_RED   = (0,0,255)
BGR_WHITE = (255,255,255)
BGR_CYAN  = (255,255,0)
BGR_BLACK = (0,0,0)

CV2_DUPLEX = cv2.FONT_HERSHEY_DUPLEX

CAMERA_INDEX = 0

IMAGE_FORMAT = '.jpg'

LOAD_OPTION = 'IMAGE'

UI_PATH = 'ui/frontalface.xml'