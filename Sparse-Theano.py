import numpy as np
import theano.tensor as T
from theano import function, shared

def main():
    X = T.dmatrix()
    k = T.dscalar()
    lambdaC = T.dscalar()
    D = shared(np.random.random((k, X.shape[1])))
    h = shared(np.random.random((X.shape[0], k)))
    gen = np.dot(h, D)
    cost = np.sum(np.square(X - np.dot(h, D))) + lambdaC * np.sum(np.log(1 + np.square(h)))
    costFunc = function([X, k, lambdaC], [cost], updates = [(D, T.grad(cost, D)), (h, T.grad(cost, h))])
    genFunc = function([], gen)

    Xvar = np.random.random((50, 30))
    kvar = 40
    lambdaCvar = 0.1
    for i in range(200):
        costVar = costFunc(Xvar, kvar, lambdaCvar)
        if i%20 == 0:
            print(costVar)
    print(genFunc())

if __name__ == '__main__':
    main()


