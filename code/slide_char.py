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
    def __init__(self, w, h, K):
        '''
        Parameters
        ----------
        (w,h): size of the window
        '''
        self.width = w
        self.height = h
        self.K = K
        self.model = []
        from sklearn import svm
        for i in range(self.K):
            self.model.append(svm.SVC(probability=True))

    def fit(self, X, y, pix=False):
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
            if  not pix:
                X_feat.append(hog(im_r,4).reshape((-1,)))
            else:
                im_r = im_r.convert('L')
                im_a = np.array(im_r.getdata()).reshape((-1,))
                X_feat.append(im_a)
        
        for i in range(self.K):
            y_feat = (y==i)
            self.model[i].fit(X_feat, y_feat)
        self.AR = np.array([[np.mean(AR[i]),np.var(AR[i])] 
                             for i in range(self.K)])

    def test(self, X, y, pix=False):
        '''
        Test model on the observation (X,y)
        '''
        y = np.array(y)
        X_feat = []
        AR = []
        for i, im in enumerate(X):
            ai = im.size[0]*1./im.size[1]
            AR.append(ai)
            im_r = im.resize((22,20))
            if not pix:
                X_feat.append(hog(im_r,4).reshape((-1,)))
            else:
                im_r = im_r.convert('L')
                im_a = np.array(im_r.getdata()).reshape((-1,))
                X_feat.append(im_a)

        p = []
        for k in range(self.K):
            p.append(self.model[k].predict_proba(X_feat)[:,1])
        p = np.array(p)
        i0 = p.argmax(axis=0)
        AR = np.array(AR)

        detect = np.exp(-(self.AR[i0,0]-AR[i0])**2/(2*self.AR[i0,1]))
        GS = np.multiply(p[i0, range(y.shape[0])], detect)

        err2 = ((y-i0).nonzero()[0].shape[0])*1./y.shape[0]
        
        err = 1- (GS>0.1).mean()
        print ('\nThis model miss {:6.2%} of the'
               ' character in the test db').format(err)

        print ('This model failed to recognize {:6.2%} character '
               'of the test db').format(err2)


    def detection(self, im, th, th1=0.1, pix=False):
        '''
        Perform the sliding window detection on im
        return a list with (x, y, class, proba)

        '''
        w = self.width
        h = self.height
        a = w*1./h
        w0 = im.size[0] - w
        h0 = im.size[1] - h
        res = []
        Ntot = w0*h0
        
        for i in range(w0):
            for j in range(h0):
                sys.stdout.write('\rChar detect...{:6.2%}'.format(
                                    (i*h0+j)*1./Ntot))
                sys.stdout.flush()
                X = im.crop((i,j,i+w,j+h))
                X_r = X.resize((22,20))
                X_feat = []
                if not pix:
                    X_feat.append(hog(X_r,4).reshape((-1,)))
                else:
                    im_r = im_r.convert('L')
                    im_a = np.array(X_r.getdata()).reshape((-1,))
                    X_feat.append(im_a)
                
                p = []
                for k in range(self.K):
                    p.append(self.model[k].predict_proba(X_feat)[0,1])
                cj = np.argmax(p)
                muj,sigj = self.AR[cj]
                GS = p[cj]*np.exp(-(muj-a)**2/(2*sigj))
                if GS > th1:
                    res.append((i,j,p, cj, p[cj], GS))
        print '\rChar detect...done  '
        return self.NMS(res, th)

    def NMS(self, res, th):
        '''
        Fusion of the patch that overlap more than th
        '''
        w = self.width
        h = self.height
        res2 = []
        pmax = [r[4] for r in res]
        while len(res)!=0:
            i0 = np.argmax(pmax)
            c = res[i0]
            del res[i0]
            del pmax[i0]
            l = [c[:2]]
            i = 0
            while i < len(res):
                c2 = res[i]
                intersec = (w-min(w,abs(c2[0]-c[0])))
                intersec *= (h-min(h,abs(c2[1]-c[1])))
                criterion = intersec / (2.*h*w - intersec)
                if criterion > th and c[3] == c2[3]:
                    l.append(c2[:2])
                    del res[i]
                    del pmax[i]
                else:
                    i+=1
            xy = np.mean(l, axis=0)
            res2.append(((xy[0], xy[1]), c[2], c[5], (w,h) ))
            
        return res2 
        
if __name__== '__main__':
    from sklearn import svm
    from sklearn.preprocessing import LabelEncoder

    import os
    import Image
    import joblib

    w = 15
    h = 17

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

