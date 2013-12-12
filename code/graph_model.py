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
        self.prior = np.zeros((K+1,K+1))

    def fit(self, mapc, w,h, th, enc):
        '''
        create a model for the decision fo the word
        '''
        import opengm
        N = len(mapc)
        numLabel = [self.K+1]*N
        self.gm = opengm.graphicalModel(numLabel)

        i0 = np.argsort(mapc, axis=0)[:,0]
        v =[]
        unary = []
        for i in i0:
            p = mapc[i][1]
            Eu = []
            for k in range(self.K):
                Eu.append(1-p[k])
            Eu.append(mapc[i][2])
            v.append([mapc[i][0], Eu])
            unary.append(Eu)
        self.vertices = v

        unary = np.array(unary)
        assert(unary.shape==(N,self.K+1))
        fid = self.gm.addFunctions(unary)
        vis = np.arange(0, N, dtype=np.uint64)
        self.gm.addFactors(fid, vis)
        
        f = opengm.PythonFunction(function=self.pairwiseE, shape=[K+1,K+1])
        fid = gm.addFunction(f)

        self.edges = []
        self.overlap = np.zeros((N,N))
        for i,v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices[i+1:]):
                dx = abs(v2[0][0]-v1[0][0])
                dy = abs(v2[0][1]-v1[0][1])
                if dx < th*w and dy < th*h:
                    
                    intersec = (h-min(h,abs(v2[0][0]-v1[0][0])))
                    intersec *= (w-min(w,abs(v2[0][1]-v1[0][1])))
                    v0 = self.lambda0*np.exp(-(100-intersec)**2)
                    BinaryE = np.ones((self.K+1,self.K+1))*v0
                    BinaryE += self.prior
                    BinaryE[self.K,self.K] = 0
                    
                    fid = self.gm.addFunction(BinaryE)
                    self.edges.append((i,j,intersec))
                    self.gm.addFactor(fid, [i,j])

                elif dx > th*w:
                    break
        
        

    def CRF_enregy(self, word):
        CRFE = 0
        for i, c in enumerate(word):
            CRFE += self.vertice[i][1][c]
        for e in self.edges:
            CRFE += self.pairwiseE(w[e[0]], w[e[1]])
        return CRFE

    def pairwiseE(self, c1, c2):
        if c1 == self.K and c2 == self.K:
            return 0
        E = self.lambda0 * np.exp(-(100-self.overlap)**2)
        if c1 == self.K or c2 == self.K:
            return E
        return E + self.prior[c1*62+c2]


    def prior_bg(self, vocabulary, enc, lambda_l=2):
        freq = np.zeros((self.K+1,self.K+1)
        n = 0.
        for w in vocabulary:
            for i in range(len(w)-1):
                c1 = enc.inverse_transform(w[i])
                c2 = enc.invers_transform(w[i+1])
                freq[c1][c2] += 1
                n += 1.
        for i in range(self.K):
            for j in range(self.K):
                self.prior[i][j] = lambda_l*(1-freq[i][j]/n)
        return self.prior


    def prior_ns(self, vocabulary, lambda_l=2):
        print 'TODO: Node specific prior' 


    def predict(self):
        import opengm
        algo = opengm.inference.TrwsExternal(self.gm)
        algo.infer()
        return algo.arg()


if __name__=='__main__':
    model = GraphicalModel()
    test = model.prior_bg(['bababa', 'baaaa', 'baaba'])
