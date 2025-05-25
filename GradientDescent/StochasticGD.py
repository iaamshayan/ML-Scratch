class SGDRegressor:
    '''
| **Aspect**            | **Explanation**                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Gradient Type**     | **Stochastic Gradient Descent (SGD)** — updates weights using **1 sample at a time**.                              |
| **Shuffling**         | Each epoch shuffles data to avoid bias from sample order. This improves convergence and prevents cyclic patterns.  |
| **Loss Function**     | Based on **Mean Squared Error (MSE)**, but updates per sample.                                                     |
| **Gradient Update**   | Per-sample: `θ = θ - α * ∇J(θ)` where `∇J` is computed using a single data point.                                  |
| **Speed**             | Faster updates than batch GD, especially beneficial for large datasets.                                            |
| **Noise**             | Updates are noisy (non-smooth), but this helps escape shallow local minima and saddle points.                      |
| **Initialization**    | Coefficients initialized to 1s — acceptable, though 0s are also commonly used.                                     |
| **Convergence**       | No convergence condition — it just runs `n_iter` epochs. Could be improved with early stopping or loss monitoring. |
| **Memory Efficiency** | Very memory-efficient — processes only one sample at a time. Good for streaming/big data scenarios.                |

    '''
    def __init__(self, learning_rate=0.01, n_iter=1000):
        # Learning rate controls how big a step we take in each update
        self.learning_rate = learning_rate

        # Number of full passes (epochs) over the training dataset
        self.n_iter = n_iter

        # Weight vector (coefficients) for each feature
        self.coef_ = None

        # Bias term (intercept)
        self.intercept_ = None

        # Track loss per iteration
        self.loss_history = []

    def fit(self, X_train, y_train):
        """
        Fit the linear regression model using **Stochastic Gradient Descent** (SGD).
        Updates weights one sample at a time.
        """

        # Initialize the intercept to 0 and coefficients to 1
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])  # Shape: (n_features,)

        for _ in range(self.n_iter):
            # Generate shuffled indices for stochasticity
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)  # Important to avoid learning bias from fixed order

            for i in indices:
                # Predict the output for one sample
                y_hat = self.intercept_ + np.dot(X_train[i], self.coef_)

                # Compute the prediction error for one sample
                error = y_train[i] - y_hat

                # Compute gradient of loss w.r.t. intercept (scalar)
                intercept_der = -2 * error

                # Update intercept
                self.intercept_ -= self.learning_rate * intercept_der

                # Compute gradient of loss w.r.t. weights
                coef_der = -2 * error * X_train[i]  # Element-wise since X_train[i] is a row vector

                # Update weights
                self.coef_ -= self.learning_rate * coef_der

            # Compute and store loss (Mean Squared Error) use any other metric if you want to
            self.loss_history.append(np.mean((y_train - y_hat) ** 2))

        # Uncomment to debug or monitor convergence
        # print(self.intercept_, '\n\n', self.coef_)

    def predict(self, X_test):
        """
        Predict target values for test data using learned weights.
        """
        return np.dot(X_test, self.coef_) + self.intercept_
