import numpy as np

class sliding_window(object):
    '''
    Sliding windows algorithm

    return a card of detection X(
    '''
    def _init_(self, w, h, model):
        '''
        Parameters
        ----------
        (w,h): size of the window
        model: model of detection
        '''
        self.width = w
        self.height = h
        self.model = model

    def fit(self, X, y):
        '''
        Fit model to the observation (X,y) of detection
        '''
        self.model.fit(X,y)

    def detection(self, th):
        pass

    def NMS(self, th):
        pass


