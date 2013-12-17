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
parser.add_argument('-s', type=float, default=0.3,
                    help='image scale')
parser.add_argument('-t', action='store_true',
                    help='Compute the test error of the SVM')
parser.add_argument('--dbchar', type=str, default='74k',
                    help='char database, 74k default, icdar ou 74icdar')
parser.add_argument('--pix', action='store_true',
                    help='use pixel instead of hog features')
args = parser.parse_args()

base_dir = '../data/'

#Load class encoder
from sklearn.preprocessing import LabelEncoder
lab = np.load(jp(base_dir, '{}_lab_char.npy'.format(args.dbchar)))
enc = LabelEncoder()
enc.fit(lab)

from slide_char import SlidingWindow
slide = SlidingWindow(1,1, 62)

#If the model doesn't exist
#Compute the character detection model
feat = 'pix' if args.pix else 'hog'
if args.r or not os.path.exists(jp(base_dir,'model/model_{}_{}.pickle'.format(
                        args.dbchar, feat))):
    filenames = np.load(jp(base_dir, '{}_list_char.npy'.format(args.dbchar)))
    lab = np.load(jp(base_dir, '{}_lab_char.npy'.format(args.dbchar)))
    y = enc.transform(lab)
    X = []
    N = filenames.shape[0]
    i = 0.
    for f in filenames:
        sys.stdout.write('\rLoad char...{:6.2%}'.format(i/N))
        sys.stdout.flush()
        i += 1
        im = Image.open(f)
        im = im.convert('RGB')
        s2 = 40./im.size[0]
        X.append(im.resize([int(s2*s) for s in im.size]))
        del im
    '''X_a_neg = np.load(jp(base_dir,'neg/neg.npy'))
    X_neg = [Image.fromarray(im, 'RGB') for im in X_a_neg]
    X.extend(X_neg)
    y = np.concatenate((y, [62]*len(X_neg)))'''
    slide.fit(X,y, pix=args.pix)
    print '\r\aCompute char model... done'
    joblib.dump(slide.model, jp(base_dir, 'model/model_{}_{}.pickle'.format(args.dbchar, feat)))
    joblib.dump(slide.AR, jp(base_dir, 'model/AR_{}_{}.pickle'.format(args.dbchar, feat)))

#Load the model
slide.model = joblib.load(jp(base_dir,'model/model_{}_{}.pickle'.format(args.dbchar, feat)))
slide.AR = joblib.load(jp(base_dir, 'model/AR_{}_{}.pickle'.format(args.dbchar, feat)))

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
    print '\rLoad char...  done'
    slide.test(X_tst,y_tst, pix=args.pix)
    sys.stdout.write('\a')
    sys.stdout.flush()
    sys.exit()

window = []

im = Image.open(jp(args.dir, args.im))
im = im.resize([int(args.s*s) for s in im.size])

def res_th(res,th):
    X = []
    for r in res:
        if r[2] > th:
            X.append(r)
    return X

'''
for w in range(10,20,2):
    for h in range(25,30,1):
        print 'window size : (', w, ',', h , ')'
        slide.width = w
        slide.height = h
        res = slide.detection(im, 0.1, 0.2, pix=args.pix)
        res2 = res_th(res, 0.4)
        window.extend(res2)
'''

w = 16
h = 25
slide.width = w
slide.height = h
res = slide.detection(im, 0.1, 0.2, pix=args.pix)
window = res_th(res, 0.6)

np.save(args.im, window)

from utils import display_char
display_char(window, im)

from graph_model import GraphicalModel

import xml.etree.ElementTree as ET
doc = ET.parse('../data/word.xml')
root = doc.getroot()
words = []

'''
for child in root:
    words.append(child.get('tag'))
'''
words.extend(['PUFF']*2000)
gm = GraphicalModel(62)
gm.prior_bg(words, enc, 1)
gm.fit(window, 1.4, enc, 0.1)

valu, val = gm.predict()

word = ''
for v in val:
    if v != 62:
        word += enc.inverse_transform(v)

print '\a', word



