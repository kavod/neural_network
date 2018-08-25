import numpy as np

class NNK(object):
    def regression(self,x,y=None,eta=1,mode='normal'):
        if mode == 'prime':
            return eta
        if mode == 'error':
            if y is None:
                print("Erreur")
                exit(0)
            return np.square(y - self.activation(x))/2
        if mode == 'corr':
            if y is None:
                print("Erreur")
                exit(0)
            return -eta*(y-self.activation(x))*x
        return x
    
    def signoid(self,x,y=None,eta=1,mode='normal'):
        if mode == 'prime':
            return eta*x*(1-x)
        if mode == 'error':
            if y is None:
                print("Erreur")
                exit(0)
            return -y *np.log(self.activation(x)) - (1 - y) *np.log(1-self.activation(x))
        if mode == 'corr':
            if y is None:
                print("Erreur")
                exit(0)
            return -eta*(y-self.activation(x))*x
        return 1/(1+np.exp(-(x)*eta))
    
    def __init__(self,layers,seed=None,eta=1,factivation='signoid',verbose=False):
        self.functions = {
                    'signoid':self.signoid,
                    'regression':self.regression,
        }
        self.weights = []
        self.values = []
        self.deltas = []
        self.eta = eta
        self.factivation = factivation
        self.verbose = verbose
        np.random.seed(seed)
        for idx,val in enumerate(layers[:-1]):
            self.weights.append(2*np.random.random((val,layers[idx+1])) - 1)
            self.values.append(np.zeros(val))
            self.deltas.append(np.zeros(val))
        print('self.weights='+str(self.weights)) if self.verbose else 1
        self.values.append(np.zeros(layers[-1]))
        self.deltas.append(np.zeros(layers[-1]))
    
    def activation(self,x):
        if self.factivation in self.functions.keys():
            return self.functions[self.factivation](x,eta=self.eta)
        else:
            return self.functions['signoid'](x,eta=self.eta)
    
    def delta(self,x,y):
        if self.factivation in self.functions.keys():
            return self.functions[self.factivation](x,y,eta=self.eta,mode='error')
        else:
            return self.functions['signoid'](x,y,eta=self.eta,mode='error')
    
    def correction(self,x,y):
        if self.factivation in self.functions.keys():
            corr = self.functions[self.factivation](x,y,eta=self.eta,mode='corr')
        else:
            corr = self.functions['signoid'](x,y,eta=self.eta,mode='corr')
        print("Correction: " + str(corr)) if self.verbose else 1
        return corr
    
    def predict(self,valInput,verbose=False):
        self.values[0] = np.array(valInput)
        print("self.values[0]="+str(self.values[0])) if self.verbose else 1
        for idx,val in enumerate(self.values[:-1]):
            self.values[idx+1] = self.activation(np.dot(self.values[idx],self.weights[idx]))
        print("self.values="+str(self.values)) if self.verbose else 1
        return self.values[-1]
    
    def backward(self,expected):
        self.deltas[-1] = (np.array(expected) - self.values[-1]) * (self.values[-1] * (1 - self.values[-1]))
        for idx in reversed(range(1,len(self.values)-1)):
            self.deltas[idx] = self.deltas[idx+1].dot(self.weights[idx].T) * self.signoidPrime(self.values[idx])
        for idx,val in enumerate(self.weights):
            val += self.values[idx].T.dot(self.deltas[idx+1])
    
    def train(self,valInput,expected):
        self.predict(valInput)
        self.backward(expected)