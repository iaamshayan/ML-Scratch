class GDRegressor:
    '''
| **Aspect**              | **Explanation**                                                                                                                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                | Batch Gradient Descent (not Stochastic)                                                                                                                                                                         |
| **Initialization**      | Coefficients are initialized to 1s (common, though zeros are also fine). Intercept = 0.                                                                                                                         |
| **Loss Function**       | Based on **Mean Squared Error (MSE)**: `(y - y_hat)^2`                                                                                                                                                          |
| **Gradient Derivation** | Gradient for intercept: `-2 * mean(y - y_hat)` , Gradient for weights: `-2/N * dot((y - y_hat), X)`                                                                                                             |
| **Update Rule**         | `theta = theta - learning_rate * gradient` — classic GD rule                                                                                                                                                    |
| **Stopping**            | Fixed iteration count (`max_iter`) — no early stopping or convergence check                                                                                                                                     |
| **Efficiency**          | Batch GD is simple but slow on large datasets. Consider **SGD** or **Mini-batch GD** for larger data.                                                                                                           |
| **Improvements**        | Add convergence condition (e.g., if loss stops changing), Add L2 regularization (Ridge), Track and plot loss to visualize learning, Standardize features before training                                        |

    '''
    def __init__(self, learning_rate=0.01, max_iter=1000):
        # Learning rate for gradient descent step
        self.learning_rate = learning_rate

        # Number of iterations for gradient descent loop
        self.max_iter = max_iter

        # Coefficients (weights) for features
        self.coef_ = None

        # Intercept (bias term)
        self.intercept_ = None

        # Track loss per iteration
        self.loss_history = []

    def fit(self, X_train, y_train):
        """
        Fits the model using Gradient Descent on the entire training data (Batch Gradient Descent)
        """

        # Initialize intercept and coefficients
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])  # Shape: (n_features,)

        # Perform batch gradient descent
        for _ in range(self.max_iter):
            # Prediction using current weights
            y_hat = np.dot(X_train, self.coef_) + self.intercept_  # Shape: (n_samples,)

            # Compute and store loss (Mean Squared Error) use any other metric if you want to
            self.loss_history.append(np.mean((y_train - y_hat) ** 2))

            # Compute the gradient of the loss w.r.t. intercept
            intercept_der = -2 * np.mean(y_train - y_hat)  # Scalar

            # Compute the gradient of the loss w.r.t. coefficients
            coef_der = -2 * (np.dot((y_train - y_hat), X_train) / X_train.shape[0])  # Shape: (n_features,)

            # Update the intercept and coefficients using gradient descent
            self.intercept_ -= self.learning_rate * intercept_der
            self.coef_ -= self.learning_rate * coef_der

        # Uncomment to view final weights and bias
        # print(self.coef_, '\n', self.intercept_)

    def predict(self, X_test):
        """
        Predicts output for test data using learned weights
        """
        return np.dot(X_test, self.coef_) + self.intercept_
