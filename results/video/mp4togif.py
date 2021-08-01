import os, sys
import imageio
import sys
import numpy as np
import cv2

def mp4togif(mp4_path, gif_path):
    reader = imageio.get_reader(mp4_path)
    frames = []
    for i, frame in enumerate(reader):
        if i%3!= 0: continue
        frame = cv2.resize(frame, (300,200))
        frames.append(frame)

    imageio.mimsave(gif_path, frames)