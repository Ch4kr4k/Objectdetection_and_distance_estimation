import cv2
import sys
from config.coco import coco
import numpy as np
import argparse

class model():

    def __init__(self, cam):
        # variables
        self.cam = cam
        self.weight = 'model/inference.pb'
        self.config = 'config/config.pbtext'
        self.classes = ['car', 'truck', 'bus', 'person']
        self.classes_width = {
            'car' : 176,
            'truck' : 195,
            'bus' : 198,
            'person' : 30
        }
        self.focal_length = 3.9

    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        return (knownWidth * focalLength) / perWidth

    def pred(self):
        cap = cv2.VideoCapture(self.cam)
        cap.set(3, 640)
        cap.set(4, 480)
        net = cv2.dnn_DetectionModel(self.weight, self.config)
        net.setInputSize(320, 320)
        net.setInputScale(1.0/127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        nms_threshold = 0.2
        thres = .35
        while True:
            ret, frame = cap.read()

            class_ids, confs, bbox = net.detect(frame, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float,confs))
            indices = list(cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold))
            if len(class_ids) != 0:
                for i in indices:
                    if coco[class_ids[i]] in self.classes:
                        box = bbox[i]
                        x, y, w, h = box[0], box[1], box[2], box[3]
                        distance = round(self.distance_to_camera(self.classes_width[coco[class_ids[i]]], self.focal_length, w),2)
                        if distance < 1:
                            distance = f"{distance*100}cm"
                        else:
                            distance = f"{distance}m"
                        cv2.rectangle(frame, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, coco[class_ids[i]], (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                        cv2.putText(frame, distance, (box[0] + 10, box[1] - 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            
            cv2.imshow("detect_window",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def run(self):
        self.pred()


def opt():
    parser = argparse.ArgumentParser(
    prog='adas',
    description='object detection and speed estimation'
    )
    parser.add_argument(
        "--cam", "-c", help="camera number 0 for built in web cam 0 or 1 for external webcam"
        )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = opt()
    cam = int(opt.cam)
    object_ = model(cam)
    object_.run()