import os
import json
from collections import OrderedDict

import numpy as np
import cv2


class MouseEventManager:
    def __init__(self, input_name):
        self.mouseEvent = {'x': None, 'y': None, 'event': None, 'flags': None,}
        cv2.setMouseCallback(input_name, self.__CallBackFunc, None)

    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        self.mouseEvent['x'] = x
        self.mouseEvent['y'] = y
        self.mouseEvent['event'] = eventType
        self.mouseEvent['flags'] = flags

    def get_coordinates(self):
        return (self.mouseEvent['x'], self.mouseEvent['y'])

    def get_event(self):
        return self.mouseEvent['event']

        
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

image_dir = './image'
images = os.listdir(image_dir)
print('images', images)

filename = []
width, height, depth = [],[],[]
xmin, ymin, xmax, ymax = [],[],[],[]
for image in images:
    print(str(image))
    if image.find('.DS_Store') >= 0:
        continue
    else:
        img = cv2.imread(os.path.join(image_dir, image))
        print(img.shape)
        shape = img.shape
        width.append(shape[0])
        height.append(shape[1])
        depth.append(shape[2])
        cv2.namedWindow('image')
        # cv2.setMouseCallback('image', learning_bbox)
        mouse_param = MouseEventManager('image')
        
        while(True):
            cv2.imshow('image', img)
            if mouse_param.get_event() == cv2.EVENT_LBUTTONDOWN:
                first_x, first_y = mouse_param.get_coordinates()
                print('min_x, min_y :', first_x, first_y)
                xmin.append(first_x)
                ymin.append(first_y)
            elif mouse_param.get_event() == cv2.EVENT_LBUTTONUP:
                second_x, second_y = mouse_param.get_coordinates()
                print('max_x, max_y :', second_x, second_y)
                xmax.append(second_x)
                ymax.append(second_y)
            if cv2.waitKey(0) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

print('height ¥n', height)
print('width ¥n', width)
print('depth ¥n', depth)
print('xmin ¥n', xmin)
print('ymin ¥n', ymin)
print('xmax ¥n', xmax)
print('ymax ¥n', ymax)

