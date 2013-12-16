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

    def fit(self, mapc, th, enc, lambda0):
        '''
        create a model for the decision fo the word
        '''
        import opengm
        N = len(mapc)
        K = self.K
        numLabel = [K+1]*N
        self.gm = opengm.graphicalModel(numLabel)

        i0 = np.argsort(mapc, axis=0)[:,0]
        self.indice = i0
        v =[]
        unary = []
        for i in i0:
            p = mapc[i][1]
            Eu = []
            for k in range(self.K):
                Eu.append(1-p[k])
            Eu.append(mapc[i][2])
            v.append([mapc[i][0], Eu, mapc[i][3]])
            unary.append(Eu)
        self.vertices = v

        unary = np.array(unary)
        assert(unary.shape==(N,K+1))
        fid = self.gm.addFunctions(unary)
        vis = np.arange(0, N, dtype=np.uint64)
        self.gm.addFactors(fid, vis)
        
        self.edges = []
        self.overlap = np.zeros((N,N))
        for i,v1 in enumerate(self.vertices):
            for j, v2 in enumerate(self.vertices[i+1:]):
                dx = abs(v2[0][0]-v1[0][0])
                dy = abs(v2[0][1]-v1[0][1])
                w = v1[2][0]
                h = v1[2][1]
                if dx < th*w and dy < th*h:
                    intersec = (w-min(w,abs(v2[0][0]-v1[0][0])))
                    intersec *= (h-min(h,abs(v2[0][1]-v1[0][1])))
                    intersec *= 100./(w*h)

                    v0 = lambda0*np.exp(-(100-intersec)**2)
                    BinaryE = np.ones((K+1, K+1))*v0
                    BinaryE += self.prior
                    BinaryE[K, K] = 0
                    
                    fid = self.gm.addFunction(BinaryE)
                    self.edges.append((i, j+i+1, intersec))
                    self.gm.addFactor(fid, [i, j+i+1])

                #elif dx > th*w:
                #    break
        

    def prior_bg(self, vocabulary, enc, lambda_l=2):
        freq = np.zeros((self.K+1,self.K+1))
        n = 0.
        for w in vocabulary:
            if len(w) > 2:
                for i in range(len(w)-1):
                    if w[i] not in ['(',')','!','&','?',
                                    '.','"',"'",'-',':',
                                    u'\xa3', u'\xc9',',',
                                    u'\xd1', u'\xe9']:
                        if w[i+1] not in ['(',')','!','&','?',
                                    '.','"',"'",'-',':',
                                    u'\xa3', u'\xc9',',',
                                    u'\xd1', u'\xe9']:
                            c1 = enc.transform(w[i])
                            c2 = enc.transform(w[i+1])
                            freq[c1][c2] += 1
                            n += 1.
        for i in range(self.K):
            for j in range(self.K):
                self.prior[i][j] = lambda_l*(1-freq[i][j]/n)
        return self.prior


    def prior_ns(self, vocabulary, lambda_l=2):
        print 'TODO: Node specific prior' 


    def predict(self, step=100):
        import opengm
        parameter = opengm.InfParam(steps=step)
        algo = opengm.inference.TrwsExternal(self.gm, parameter=parameter)
        algo.infer()
        i0 = self.indice.argsort()
        val = np.array(algo.arg())
        return val[i0]


if __name__=='__main__':
    model = GraphicalModel()
    test = model.prior_bg(['bababa', 'baaaa', 'baaba'])
