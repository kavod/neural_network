import numpy as np

class NNK(object):
    def __init__(self,layers,taux=1,aInput=1,bInput=0,aOutput=1,bOutput=0,factivation='signoid'):
        self.weights = []
        self.values = []
        self.deltas = []
        self.expected = []
        self.taux = taux
        self.aInput = aInput
        self.bInput = bInput
        self.aOutput = aOutput
        self.bOutput = bOutput
        self.factivation = factivation
        for idx,val in enumerate(layers[:-1]):
            #self.weights.append(2*np.random.random((val,layers[idx+1])) - 1)
            self.weights.append(np.ones((val,layers[idx+1])) /2)
            self.values.append(np.zeros(val))
            self.expected.append(np.zeros(val))
            self.deltas.append(np.zeros(val))
        self.values.append(np.zeros(layers[-1]))
        self.expected.append(np.zeros(layers[-1]))
        self.deltas.append(np.zeros(layers[-1]))
    
    def activation(self,x):
        if self.factivation == 'signoid':
            return self.signoid(x)
        else:
            return self.signoid(x)
    
    def predict(self,valInput,verbose=False):
        self.values[0] = np.array(valInput)*self.aInput + self.bInput
        for idx,val in enumerate(self.values[:-1]):
            print(str(idx+1)+":"+str(self.values[idx])) if verbose else True
            self.values[idx+1] = self.activation(np.dot(self.values[idx],self.weights[idx]))
        return (self.values[-1]-self.bOutput)/self.aOutput
    
    def signoid(self,x):
        return 1/(1+np.exp(-(x)*self.taux))
    
    def signoidPrime(self,x):
        return x * (1 - x)
    
    def backward(self,expected):
        #print("Weights:"+str(self.weights))
        #print("Deltas:"+str(self.deltas))
        self.deltas[-1] = (np.array(expected) - self.values[-1]) * (self.values[-1] * (1 - self.values[-1]))
        for idx in reversed(range(1,len(self.values)-1)):
            #print(self.deltas[idx])
            self.deltas[idx] = self.deltas[idx+1].dot(self.weights[idx].T) * self.signoidPrime(self.values[idx])*self.taux
        for idx,val in enumerate(self.weights):
            val += self.values[idx].T.dot(self.deltas[idx+1])
    
    def train(self,valInput,expected):
        self.predict(valInput)
        self.backward(expected)
        