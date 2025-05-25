class LR:
    '''
    Replicating Linear Regression from sklearn using Ordinary Least Squares (OLS) method.
    Only supports simple linear regression (one feature).
    '''

    def __init__(self):
        # m is the slope (coefficient), b is the intercept
        self.m = None
        self.b = None

    def train(self, X_train, y_train):
        '''
        Fits the linear regression model to the training data.
        Parameters:
            X_train: numpy array of shape (n_samples,) - Feature values
            y_train: numpy array of shape (n_samples,) - Target values
        '''

        # Center the data by subtracting the mean (helps with numerical stability)
        y_centered = y_train - y_train.mean()
        X_centered = X_train - X_train.mean()

        # Compute slope (m) using the closed-form solution of OLS:
        # m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)^2)
        self.m = X_centered.dot(y_centered) / X_centered.dot(X_centered)

        # Compute intercept (b) using the formula:
        # b = ȳ - m * x̄
        self.b = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        '''
        Predicts the target values for the test input.
        Parameters:
            X_test: numpy array of shape (n_samples,)
        Returns:
            Predicted values as a numpy array
        '''
        return (self.m * X_test) + self.b
