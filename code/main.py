import numpy as np
import sys
import Image

import os
import joblib
from os.path import join as jp

import argparse
parser = argparse.ArgumentParser(description='Street text detection')
parser.add_argument('-r', action='store_true', help='Recompute model')
parser.add_argument('--im', type=str, help='Image to detect',
                    default='00_00.jpg')
parser.add_argument('--dir', type=str, help='Image base folder',
                    default='../data/svt1/img')
parser.add_argument('-c', type=int, default=10,
                    help='character display')
parser.add_argument('--width', type=int, default=200)
parser.add_argument('--height', type=int, default=200)
args = parser.parse_args()

base_dir = '../data/'

#Load class encoder
from sklearn.preprocessing import LabelEncoder
lab = np.load(jp(base_dir, 'char/lab_char.npy'))
enc = LabelEncoder()
enc.fit(lab)

w = h = 15

from slide_char import SlidingWindow
slide = SlidingWindow(w, h)

#If the model doesn't exist
#Compute the character detection model
if args.r or not os.path.exists(jp(base_dir,'model/model.pickle')):
    filenames = np.load(jp(base_dir, 'char/list_char.npy'))
    y = enc.transform(lab)
    X = []
    for f in filenames:
        im = Image.open(f)
        X.append(im.resize([int(0.3*s) for s in im.size]))
        del im
    slide.fit(X,y)
    joblib.dump(slide.model, jp(base_dir, 'model/model.pickle'))
    joblib.dump(slide.AR, jp(base_dir, 'model/AR.pickle'))

#Load the model
slide.model = joblib.load(jp(base_dir,'model/model.pickle'))
slide.AR = joblib.load(jp(base_dir, 'model/AR.pickle'))

im = Image.open(jp(args.dir, args.im))
im = im.resize((args.width, args.height))
res = slide.detection(im, 0.05)

from utils import display_char
display_char(res, args.c, w, h, im)
