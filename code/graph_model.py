import numpy as np

class GraphicalModel(object):
    '''
    Graphical model used for the word decision

    return the detected word

    '''
    def __init__(self, K):
        '''
        Constructeur

        epsilon = '.'


        self.vertices => list de sommet
            -Eu[c] -> confidence svm
        self.edges => list de cote
            -overlap -> overlap des vertices
        '''
        self.K = K

    def fit(self, mapc, w,h, th, enc):
        '''
        create a model for the decision fo the word
        '''
        l = 0
        for m in mapc:
            l += len(m)
        i = 0
        v = []
        while i < l:
            xy = mapc[0][0][0]
            km = 0
            for k in range(1,K):
                if xy > mapc[k][0][0]:
                    xy = mapc[k][0][0]
                    km = k
            p = mapc[km][0][1]
            Eu = {}
            for i in range(len(p)):
                Eu[enc.inverse_transform(i)] = 1-p[i]
            Eu['.'] = mapc[km][0][2]
            v.append([xy, Eu])
            del mapc[km][0]
        self.vertices = v

        self.edges
        for i,v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices[i+1:]):
                dx = abs(v2[0][0]-v1[0][0])
                dy = abs(v2[0][1]-v1[0][1])
                if dx < th*w and dy < th*h:
                    intersec = (h-min(h,abs(v2[0][0]-v1[0][0])))
                    intersec *= (w-min(w,abs(v2[0][1]-v1[0][1])))
                    self.edges.append((i,j,intersec))
                
    
    def CRF_enregy(self, word):
        CRFE = 0
        for i, c in enumerate(word):
            CRFE += self.vertice[i][1][c]
        for e in self.edges:
            CRFE += self.pairwiseE(e[2], w[e[0]], w[e[1]])
        return CRFE

    def pairwiseE(self, overlap, c1, c2):
        if c1 == '.' and c2 == '.':
            return 0
        E = self.lambda0 * np.exp(-(100-overlap)**2)
        if c1 == '.' or c2 == '.':
            return E
        return E + self.prior[c1+c2]


    def prior_bg(self, vocabulary, lambda_l=2):
        freq = {}
        n = 0.
        for w in vocabulary:
            for i in range(len(w)-1):
                c = w[i:i+2]
                print c
                freq[c] = freq.get(c,0)+1
                n += 1.
        E = {}
        for k in freq.keys():
            freq[k] /= n
            E[k] = lambda_l*(1-freq[k])
        self.prior = E
        return E 


    def prior_ns(self, vocabulary, lambda_l=2):
        print 'TODO: Node specific prior' 





if __name__=='__main__':
    model = GraphicalModel()
    test = model.prior_bg(['bababa', 'baaaa', 'baaba'])
