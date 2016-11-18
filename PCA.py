"""
    PCA implementation using Numpy -  By dhruvramani
    np.cov() => computes covariance matrix returning dimensions col x col
    np.linalg.eig() => computes eignValues and eignVectors, I sort them
    eg. 
        X : 30 x 50
        sigma : 50 X 50,          U : 50 X 50
        U[:, :20].T : 20 x 50 ,   X.T : 50 x 30 
        After np.dot : 20 x 30 => Hence take transpose
"""
import numpy as np

def getData():
    return np.random.random((20,30))

class PCA:
    def __init__(self, X):
        self.X = X # - np.mean(X)
        self.Xdash = None

    def train(self,k):
        sigma = np.cov(x, rowvar=False) #(1/self.X.shape[0]) * (np.dot(self.X.T, self.X))
        v, U = np.linalg.eig(sigma)
        idx = np.argsort(v)[::-1]
        U = U[:,idx] #= np.asarray([pairs[i][1] for i in range(len(pairs))])
        self.Xdash = np.dot(U[:, :k].T, self.X.T).T

    def getRep(self):
        return self.Xdash

    def getK(self,eignValues, n):
        for i in range(1, n):
            if(np.sum(eignValues[0:i,:]) / np.sum(eignValues) >= 0.99)
                return i
        return n/2

def main():
    k = 30
    data = getData()
    p = PCA(data)
    p.train(k) 
    print(p.getRep())

if __name__ == '__main__':
    main()