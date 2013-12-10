import sys
import numpy as np

import pyximport
pyximport.install()
from features import hog
from scipy.misc import imresize, imread


class SlidingWindow(object):
    '''
    Sliding windows algorithm

    return a card of detection X
    '''
    def __init__(self, w, h):
        '''
        Parameters
        ----------
        (w,h): size of the window
        '''
        self.width = w
        self.height = h
        self.K = 61
        self.model = []
        from sklearn import svm
        for i in range(self.K):
            self.model.append(svm.SVC(probability=True))

    def fit(self, X, y):
        '''
        Fit model to the observation (X,y) of detection
        '''
        y = np.array(y)
        X_feat = []
        AR = [[] for i in range(self.K)]
        for i, im in enumerate(X):
            ai = im.size[0]*1./im.size[1]
            AR[y[i]].append(ai)
            im_r = im.resize((22,20))
            X_feat.append(hog(im_r,4).reshape((-1,)))
        
        for i in range(self.K):
            y_feat = (y==i)
            self.model[i].fit(X_feat, y_feat)
        self.AR = np.array([[np.mean(AR[i]),np.var(AR[i])] for i in
        range(self.K)])

    def detection(self, im, th, th1=0.1):
        '''
        Perform the sliding window detection on im
        return a list with (x, y, class, proba)

        '''
        w = self.width
        h = self.height
        a = w/h
        w0 = im.size[0] - w
        h0 = im.size[1] - h
        res = [[] for i in range(self.K)]
        Ntot = w0*h0
        
        for i in range(w0):
            for j in range(h0):
                sys.stdout.write('\rChar detect...{:6.2%}'.format(
                                    (i*h0+j)*1./Ntot))
                sys.stdout.flush()
                X = im.crop((i,j,i+w,j+h))
                X_r = X.resize((22,20))
                X_feat = hog(X_r,4).reshape((1,-1))
                p = []
                for k in range(self.K):
                    p.append(self.model[k].predict_proba(X_feat)[0,1])
                cj = np.argmax(p)
                muj,sigj = self.AR[cj]
                GS = p[cj]*np.exp(-(muj-a)**2/(2*sigj))
                if GS > th1:
                    for k in range(self.K):
                        res[k].append((i,j,p[k]))
        print '\rChar detect...done  '
        res2 = []
        for k in range(self.K):
            sys.stdout.write('\rNMS...{:6.2%}'.format(k*1./self.K))
            sys.stdout.flush()
            res2.append(self.NMS(res[k], th))
        print '\rNMS...done  '
        return res2

    def NMS(self, res, th):
        '''
        Fusion of the patch that overlap more than th
        '''
        w = self.width
        h = self.height
        r = []
        while len(res)!=0:
            i0 = np.argmax(res, axis=0)[2]
            c = res[i0]
            del res[i0]
            l = [c[:2]]
            i = 0
            while i < len(res):
                c2 = res[i]
                intersec = (h-min(h,abs(c2[0]-c[0])))
                intersec *= (w-min(w,abs(c2[1]-c[1])))
                criterion = intersec / (2.*h*w - intersec)
                if criterion > th:
                    #l.append(c2[:2])
                    del res[i]
                else:
                    i+=1
            r.append((np.mean(l,axis=0), c[2]))
            
        return r        


if __name__== '__main__':
    from sklearn import svm
    from sklearn.preprocessing import LabelEncoder

    import os
    import Image
    import joblib

    w = 15
    h = 20

    test = SlidingWindow(w, h)
    if not os.path.exists('../data/model/model.pickle'):

        filenames = np.load('../data/char/list_char.npy')
        labels = np.load('../data/char/lab_char.npy')
        enc = LabelEncoder()
        y = enc.fit_transform(labels)
        X = []
        for f in filenames:
            im = Image.open(f)
            X.append(im.resize([int(0.3 * s) for s in im.size]))
            del im
        test.fit(X,y)
        joblib.dump(test.model, '../data/model/model.pickle')
        joblib.dump(test.AR, '../data/model/AR.pickle')
    else:
        model = joblib.load('../data/model/model.pickle')
        AR = joblib.load('../data/model/AR.pickle')
        test.model = model
        test.AR = AR
    im = Image.open('../data/svt1/img/00_00.jpg')
    im = im.resize((200,200))
    res = test.detection(im,0.1)

    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    plt.clf()
    ax = plt.gca()
    for r in res[10]:
        ax.add_patch(Rectangle(r[0],w ,h, fc='none', ec='blue'))
    ax.autoscale_view()
    plt.imshow(im)
    plt.show()

