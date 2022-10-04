import numpy as np
class KNNalgo:
    def __init__(self, n):
        self.n = n
        self.X_train = None
        self.y_train = None
        
        
    def euclidean(self,x,y):
        a=((y-x)**2)**(1/2)
        return np.sum(a)
    
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict1(self, point):
        dis = [(self.euclidean(point,i)) for i in self.X_train]
        k=np.argsort(dis)[:self.n]
        kb=[]
        for i in k:
            kb.append(self.y_train[i])
        res=max(kb,key=kb.count)
        return res
    
    def result(self,data):
        return np.array([self.predict1(i) for i in data])
    
    
    def evaluate(self, x_test, y_test):
        precision = np.sum(np.array(self.result(x_test) == y_test)) / len(y_test)
        print("accuary:%f"%(precision))
        return precision
        
