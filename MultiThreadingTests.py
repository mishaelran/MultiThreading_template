from threading import *
from time import sleep
import multiprocessing
import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import glob
import time


class Sift(Thread):

    def init_this_class(self, image_gray, sift_obj, image_num):
        """
        cant use normal __init__ because Thread has its own __init__ inside the threading
        library.
        """
        self.image_gray = image_gray
        self.sift_obj = sift_obj
        self.image_num = image_num

    def run(self):
        print(Thread.getName(self))
        self.images_gray = cv2.cvtColor(self.image_gray, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift_obj.detectAndCompute(self.image_gray, None)
        self.images_gray = cv2.drawKeypoints(self.images_gray, kp, self.images_gray,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('Debug\sift_keypoints' + str(self.image_num) + '.jpg', self.images_gray)



def main():
    start_time = time.clock()
    print("starting main:")
    check_cpu_threads = multiprocessing.cpu_count()
    print(check_cpu_threads)
    DIR = 'images\\'
    MASKS = 'masks\\'
    number_of_image_in_dir = (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    images = [cv2.imread(file) for file in glob.glob(DIR + "*")]
    images_gray = images
    kp = [None] * len(images)
    des = [None] * len(images)
    sift_obj = cv2.xfeatures2d.SIFT_create()
    objs = [Sift() for i in range(number_of_image_in_dir)]

    for i in range(number_of_image_in_dir):
        objs[i].init_this_class(images_gray[i], sift_obj, i)

    for i in range(number_of_image_in_dir):
        objs[i].start()


    print(time.clock() - start_time)

main()
