class LR:
    def __init__(self):
        self.m = None
        self.b = None

    def train(self,X_train,y_train):
        y_centered = y_train - y_train.mean()
        X_centered = X_train - X_train.mean()
        self.m = X_centered.dot(y_centered)/X_centered.dot(X_centered)
        self.b = y_train.mean() - (self.m*X_train.mean())

    def predict(self,X_test):
        return (self.m*X_test + self.b)
