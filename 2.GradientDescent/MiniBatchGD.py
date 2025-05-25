import numpy as np

class miniBatchGD:
    '''
| **Aspect**           | **Details**                                                                             |
| -------------------- | --------------------------------------------------------------------------------------- |
| **Technique**        | Mini-Batch Gradient Descent                                                             |
| **Batch Size**       | Customizable via constructor (`batch_size`)                                             |
| **Loss Function**    | Based on **Mean Squared Error (MSE)**                                                   |
| **Gradient Update**  | Uses gradient calculated from a randomly sampled mini-batch                             |
| **Efficiency**       | Balanced — less noisy than SGD, less memory intensive than full batch GD                |
| **Random Sampling**  | Uses `random.sample`, which selects samples **without replacement** for each mini-batch |
| **Weights Update**   | Per mini-batch, not per sample or full dataset                                          |
| **Shuffling**        | Only sampled per mini-batch; could be improved by full reshuffling per epoch            |
| **Learning Rate**    | Fixed — may cause issues if too high/low over many epochs                               |
| **Intercept Update** | Correctly handled separately from weight vector                                         |
    '''
    def __init__(self, batch_size=32, epochs=1000, learning_rate=0.01, tol=1e-6, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.intercept_ = None
        self.coef_ = None
        self.tol = tol                  # Tolerance for early stopping
        self.verbose = verbose          # Toggle print statements
        self.losses = []                # Store losses for visualization

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape

        # Initialize weights with small random values
        self.coef_ = np.random.randn(n_features) * 0.01
        self.intercept_ = 0

        for epoch in range(self.epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Forward pass
                y_hat = np.dot(X_batch, self.coef_) + self.intercept_
                error = y_batch - y_hat

                # Gradient calculation
                intercept_der = -2 * np.mean(error)
                coef_der = -2 * np.dot(error, X_batch) / X_batch.shape[0]

                # Update weights
                self.intercept_ -= self.learning_rate * intercept_der
                self.coef_ -= self.learning_rate * coef_der

            # Compute loss for convergence monitoring
            y_pred_all = np.dot(X_train, self.coef_) + self.intercept_
            loss = np.mean((y_train - y_pred_all) ** 2)
            self.losses.append(loss)

            # Print loss every 100 epochs
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

            # Early stopping
            if epoch > 0 and abs(self.losses[-2] - self.losses[-1]) < self.tol:
                if self.verbose:
                    print(f"Stopped early at epoch {epoch}, Loss: {loss:.6f}")
                break

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
