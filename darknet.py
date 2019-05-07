import ctypes


class BOX(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("w", ctypes.c_float),
                ("h", ctypes.c_float)]


class DETECTION(ctypes.Structure):
    _fields_ = [("bbox", BOX),
                ("classes", ctypes.c_int),
                ("prob", ctypes.POINTER(ctypes.c_float)),
                ("mask", ctypes.POINTER(ctypes.c_float)),
                ("objectness", ctypes.c_float),
                ("sort_class", ctypes.c_int)]


class IMAGE(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int),
                ("h", ctypes.c_int),
                ("c", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_float))]


class METADATA(ctypes.Structure):
    _fields_ = [("classes", ctypes.c_int),
                ("names", ctypes.POINTER(ctypes.c_char_p))]

# get_network_boxes
lib = ctypes.CDLL("resources/libraries/libdarknet.so", ctypes.RTLD_GLOBAL)
lib.network_width.argtypes = [ctypes.c_void_p]
lib.network_width.restype = ctypes.c_int
lib.network_height.argtypes = [ctypes.c_void_p]
lib.network_height.restype = ctypes.c_int

predict = lib.network_predict
predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
predict.restype = ctypes.POINTER(ctypes.c_float)

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
get_network_boxes.restype = ctypes.POINTER(DETECTION)


free_detections = lib.free_detections
free_detections.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

network_predict = lib.network_predict
network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [ctypes.c_void_p]

load_net = lib.load_network
load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
load_net.restype = ctypes.c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, ctypes.c_int, ctypes.c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [ctypes.c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [ctypes.c_void_p, IMAGE]
predict_image.restype = ctypes.POINTER(ctypes.c_float)