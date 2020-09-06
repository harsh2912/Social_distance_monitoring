from model import Model
from utils import *
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser()
# general
parser.add_argument('--video-path', help='')
parser.add_argument('--save-path')

if __name__=='__main__':
    args = parser.parse_args()
    model = Model()
    cap = cv2.VideoCapture(args.video_path)

    ret , frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(args.save_path, fourcc, 25.0, (640,480))
    
    while ret:
        frame = cv2.resize(frame,(640,480))
        out = model.get_class_outputs(frame)
        box_lst = []
        image = frame
        scores = out[0]['instances'].scores
        bboxes = out[0]['instances'].pred_boxes.tensor[scores>=0.5]
        box_lst.append(bboxes)
        boxes2color = []
        for k in range(len(box_lst[0])):
            reference =  box_lst[0][k] 
            lst_points = list(box_lst[0].clone().detach())
            lst_points.pop(k)
            color = find_color_of_box(reference,lst_points,30)
            boxes2color.append([reference,color])
        for h in range(len(boxes2color)):
            ref,col = boxes2color[h]
            image = draw_box(image,ref,col)
        image = cv2.resize(image,(640,480))
        output.write(image)

        ret,frame = cap.read()