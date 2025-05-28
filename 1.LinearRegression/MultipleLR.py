class MLR:
    """
    Multiple Linear Regression (MLR) implementation using the Normal Equation (OLS method).
    """

    def __init__(self):
        """
        Initialize model parameters.
        """
        self.coef = None          # Coefficients for features
        self.intercept = None     # Intercept term

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data using the Normal Equation.

        Parameters:
        - X_train: numpy array of shape (n_samples, n_features)
        - y_train: numpy array of shape (n_samples,)
        """
        # Add a column of ones to X_train to account for the intercept
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Calculate coefficients: (X^T * X)^-1 * X^T * y
        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Separate the intercept and the feature coefficients
        self.intercept = beta[0]
        self.coef = beta[1:]

    def predict(self, X_test):
        """
        Predict target values for test data.

        Parameters:
        - X_test: numpy array of shape (n_samples, n_features)

        Returns:
        - predictions: numpy array of shape (n_samples,)
        """
        return X_test @ self.coef + self.intercept
