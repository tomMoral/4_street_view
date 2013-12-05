
#import HOG for feature computation
import pyximport
pyximport.install()

from features import hog


'''
load images and compute features

return a numpy array X (n_item, n_features)
'''
