import numpy as np
import time
import cv2
import os
from config import config
from tracker import SiamFCTracker
import argparse
import time
from region_to_bbox import region_to_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='siamfc.pth')
arg = parser.parse_args()
model_dir = arg.model_dir


def run_camera():

    tracker = SiamFCTracker(model_dir)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    bbox = cv2.selectROI(frame, fromCenter=False)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tracker.init(frame, bbox)
    while True:
        start = time.time()
        ret, frame = cap.read()
        bbox = tracker.update(frame)
        bbox = np.asarray([bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]])
        end = time.time()
        fps = 1 / (end - start)
        print("fps:"+str(fps))
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                      (255, 0, 0), thickness=2)
        cv2.imshow('img', frame)
        if cv2.waitKey(10) == 27:
            break










if __name__ == '__main__':
    run_camera()
