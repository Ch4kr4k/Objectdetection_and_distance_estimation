import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import argparse

DetectionResult = None

class detect():
    def __init__(self, video, mode ,model = None,):
        self.video = video
        self.model = model
        self.mode = mode
        global DetectionResult
        
        # initializing settings mp for loading model
        self.BaseOptions = mp.tasks.BaseOptions
        self.DetectionResult = mp.tasks.components.containers.Detection
        self.ObjectDetector = mp.tasks.vision.ObjectDetector
        self.ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        return (knownWidth * focalLength) / perWidth

    def pred(self, frame, frame_ts):# model path
        # Creating base config for loading Model\
        if self.mode == "video":
            optionsV = self.ObjectDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=self.model),
            max_results=5,
            running_mode=self.VisionRunningMode.VIDEO)
            detector = vision.ObjectDetector.create_from_options(optionsV)
            res = detector.detect_for_video(frame, frame_ts)

        elif self.mode == "cam":
            optionsV = self.ObjectDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=self.model),
            max_results=5,
            result_callback = self.print_result,
            running_mode=self.VisionRunningMode.LIVE_STREAM)
            detector = vision.ObjectDetector.create_from_options(optionsV)
            res = detector.detect_async(frame, frame_ts)
        
        return res  # return the detected informations

    # for drawing bounding box
    def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
            print('detection result: {}'.format(result))

    def visualize(
        self,
        image,
        detection_result
    ) -> np.ndarray:  # returns a numpy arrays

        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        BOX_Colour = (0, 255, 0)
        area_colour = (150, 58, 255)
        text_colour = (0,0,255)

        list_ = ["car", "truck", "bus"]  # needed classes to detect

        rwidth = {
        "car" : 176,
        "bus" : 195,
        "truck" : 198,
        }

        focal_lenth = 3.9

        for detection in detection_result.detections:  # loop through all the detected objects
        # Draw bounding_box
            if detection.categories[0].category_name in list_: # draw only if detected object is in list
                category = detection.categories[0] # category extractions
                category_name = category.category_name # name extractions
                probability = round(category.score, 2) # extract confident scores
                if probability > .3: # set thresshold to 30%
                    bbox = detection.bounding_box  # extract bounding box informations
                    start_point = bbox.origin_x, bbox.origin_y # bounding box start point
                    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height # bounding box information
                    cv2.rectangle(image, start_point, end_point, BOX_Colour, 1) # create rectangle
                    dis = str(round(self.distance_to_camera(rwidth[category_name], focal_lenth, bbox.width),2))
                    # Draw label and score
                    result_text = category_name + ' (' + str(probability) + ')'
                    text_location = (MARGIN + bbox.origin_x,
                                     MARGIN + ROW_SIZE + bbox.origin_y+2)
                    dis_location = (MARGIN + bbox.origin_x,
                                     MARGIN + ROW_SIZE + bbox.origin_y-20)
                    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                                FONT_SIZE, text_colour, 1) # create labels
                    cv2.putText(image, dis, dis_location, cv2.FONT_HERSHEY_PLAIN,
                                FONT_SIZE, text_colour, 1) # create labels
                    
                else:
                    image = image # if frame has no car set frame to orginal frame
        
        return image #returns the frame

    def run(self):
        skip_factor = 2
        count = 0
        if self.mode == "video":
            cap = cv2.VideoCapture(self.video)
        elif self.mode == "cam":
            cap = cv2.VideoCapture(self.video)

        if not cap.isOpened():
            print("Error opening video file")
        #segmented_area = np.array([[3, 777], [1877, 795], [1916, 1007], [8, 1073]])
        #tri_area = np.array([[2, 395], [527, 892], [8, 927]])
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                count+=1
                if count % skip_factor == 0:
                    hieght, width, channel = frame.shape
                    frame = cv2.resize(frame,(width//2, hieght//2))
                    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_copy = np.copy(frame.numpy_view())
                    #frame_copy1 = np.copy(frame.numpy_view())
                    #frame_copy = cv2.fillPoly(frame_copy, pts=[segmented_area], color=(255, 0, 0))
                    #frame_copy = cv2.fillPoly(frame_copy, pts=[tri_area], color=(255, 0, 0))
                    #frame_copy = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame_copy))
                    res = self.pred(frame, frame_timestamp_ms)
                    
                    annotated_image= self.visualize(frame_copy, res)
                    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    final_frame = cv2.cvtColor(rgb_annotated_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Video', final_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break




def opt():

    parser = argparse.ArgumentParser(
        prog='adas',
        description='object detection and speed estimation'
        )
    parser.add_argument(
        "mode",default="cam", help="eg video, cam"
        )
    parser.add_argument(
        "--model", "-m", help="path/model"
        )
    parser.add_argument(
        "--video", "-v", help="path/video"
        )
    parser.add_argument(
        "--cam", "-c", help="camera number 0 for built in web cam 1 or 2 for external webcam"
        )
    args = parser.parse_args()
    return args

def main():
    args = opt()
    mode = args.mode
    mode_list = ["video", "cam"]
    if mode in mode_list:
        if args.mode == "video":
            video = args.video
        elif args.mode == "cam":
            video = int(args.cam)
        else:
            print("select a mode \n syntax python model.py video --video video.mp4 --model model")
        if args.model:
            model = args.model
        model_instance = detect(video, mode, model)
        model_instance.run()
    else:
        print("select a mode \n syntax python model.py video --video video.mp4 --model model")

if __name__ == '__main__':
    main()
