import ctypes
import time

import cv2
import darknet
import numpy

class Detector():
    def __init__(self, *args, **kwargs):

        self.__darknet = darknet
        self.__predict = None

    def convert_back(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def array_to_image(self, arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2, 0, 1)
        channels, height, width = arr.shape[0:3]
        arr = numpy.ascontiguousarray(arr.flat, dtype=numpy.float32) / 255.0
        data = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        im = self.__darknet.IMAGE(width, height, channels, data)
        return im, arr

    def detect(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        im, image = self.array_to_image(image)
        self.__darknet.rgbgr_image(im)
        num = ctypes.c_int(0)
        pnum = ctypes.pointer(num)
        self.__darknet.predict_image(net, im)
        dets = self.__darknet.get_network_boxes(net, im.w, im.h, thresh,
                                 hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms: self.__darknet.do_nms_obj(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            a = dets[j].prob[0:meta.classes]
            if any(a):
                ai = numpy.array(a).nonzero()[0]
                for i in ai:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i],
                                (b.x, b.y, b.w, b.h)))

        res = sorted(res, key=lambda x: -x[1])
        if isinstance(image, bytes): self.__darknet.free_image(im)
        self.__darknet.free_detections(dets, num)
        return res

if __name__ == "__main__":
    # add darknet library
    detector = Detector()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap.set(3, 640)
    # cap.set(4, 480)
    net = darknet.load_net(b"resources/models/eyes_detector.cfg", b"resources/models/eyes_detector.weights", 0)
    meta = darknet.load_meta(b"resources/models/eyes_detector.data")
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        start_time = time.time()
        r = detector.detect(net, meta, img)
        for i in r:
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = detector.convert_back(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 1)) + "]", (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
        cv2.imshow("img", img)

        print(time.time() - start_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break