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
parser.add_argument('-s', type=int, default=0.3,
                    help='image scale')
parser.add_argument('-t', action='store_true',
                    help='Compute the test error of the SVM')
args = parser.parse_args()

base_dir = '../data/'

#Load class encoder
from sklearn.preprocessing import LabelEncoder
lab = np.load(jp(base_dir, 'char/lab_char.npy'))
enc = LabelEncoder()
enc.fit(lab)

w = 15
h = 20

from slide_char import SlidingWindow
slide = SlidingWindow(w, h)

#If the model doesn't exist
#Compute the character detection model
if args.r or not os.path.exists(jp(base_dir,'model/model.pickle')):
    filenames = np.load(jp(base_dir, 'char/list_char.npy'))
    y = enc.transform(lab)
    X = []
    N = filenames.shape[0]
    i = 0.
    for f in filenames:
        sys.stdout.write('\rLoad char...{:6.2%}'.format(i/N))
        sys.stdout.flush()
        i += 1
        im = Image.open(f)
        s2 = 40./im.size[0]
        X.append(im.resize([int(s2*s) for s in im.size]))
        del im
    slide.fit(X,y)
    print 'Compute char model... done'
    joblib.dump(slide.model, jp(base_dir, 'model/model.pickle'))
    joblib.dump(slide.AR, jp(base_dir, 'model/AR.pickle'))

#Load the model
slide.model = joblib.load(jp(base_dir,'model/model.pickle'))
slide.AR = joblib.load(jp(base_dir, 'model/AR.pickle'))

if args.t:
    lab_tst = np.load(jp(base_dir, 'char_tst/lab_char.npy')) 
    file_tst = np.load(jp(base_dir, 'char_tst/list_char.npy'))
    y_tst = enc.transform(lab_tst)
    X_tst = []
    N = file_tst.shape[0]
    i = 0.
    for f in file_tst:
        sys.stdout.write('\rLoad char...{:6.2%}'.format(i/N))
        sys.stdout.flush()
        i += 1
        im = Image.open(f)
        s2 = 40./im.size[0]
        X_tst.append(im.resize([int(s2*s) for s in im.size]))
        del im
    slide.test(X_tst,y_tst)
    sys.exit()

im = Image.open(jp(args.dir, args.im))
im = im.resize([int(args.s*s) for s in im.size])
res = slide.detection(im, 0.1, 0.2)

from utils import display_char
display_char(res, w, h, im)

from graph_model import GraphicalModel

gm = GraphicalModel(61)
gm.fit(res, w, h, 1.15, enc)

valu = gm.predict()
