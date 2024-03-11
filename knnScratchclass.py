import numpy as np 
from statistics import mode
class KnnFromScratch:
    def __init__(self,n=5):
        self.neighbours = n
        self.X_train = None
        self.y_train = None
    def train(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self,X_test):
        y_pred = []
        for a in X_test:
            distance = []
            for b in self.X_train:
                distance.append(np.linalg.norm(a-b))
            majority = sorted(list(enumerate(distance)),key = lambda x:x[1])[0:5]
            label = mode([self.y_train[i[0]] for i in majority])
            y_pred.append(label)
        return y_pred