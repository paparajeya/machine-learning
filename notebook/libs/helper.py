import numpy as np
import matplotlib.pyplot as plt


class Helper:
    def __init__(self):
        # create a random number generator
        self.rng = np.random.default_rng()

    def create_linear_dataset(
        self,
        title: str,
        num_samples: int = 250,
        x_range: int = 500,
        x_min: int = 0,
        slope: int = 2,
        intercept: int = 3,
        noise: int = 200,
    ):
        """
        Create a linear dataset:
        Parameters:
            title: title of the dataset
            num_samples: number of samples
            x_range: range of x values
            x_min: minimum x value
            slope: slope of the linear equation
            intercept: intercept of the linear equation
            noise: noise in the dataset

        Returns:
            x: x values
            y: y values
            title: title of the dataset
        """
        # Set seed for reproducibility
        np.random.seed(self.rng.integers(0, 1000))
        # Create a random x values within the range of x_range and x_min
        x = np.random.rand(num_samples, 1) * x_range + x_min
        # Create a linear y values
        y = slope * x + intercept + np.random.randn(num_samples, 1) * noise
        # Return the x, y and title
        return (x, y, title)
    
    def plot_dataset(self, x, y, title):
        """
        Plot the dataset
        Parameters:
            x: x values
            y: y values
            title: title of the dataset
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', alpha=0.5, label="Data points")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
        
    def plot_dataset_with_hypothesis(self, x, y, y_pred, slope, intercept, title):
        """
        Plot the dataset with hypothesis
        Parameters:
            x: x values
            y: y values
            title: title of the dataset
            hypothesis: hypothesis
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', alpha=0.5, label="Data points")
        plt.plot(x, y_pred, color='red', label=f"Hypothesis: y = {slope:.2f}x + {intercept:.2f}") 
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        Calculate the mean absolute error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            mean absolute error
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate the mean squared error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def r_squared(self, y_true, y_pred):
        """
        Calculate the R-squared value
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            R-squared value
        """
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (numerator / denominator)
    
    def root_mean_squared_error(self, y_true, y_pred):
        """
        Calculate the root mean squared error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            root mean squared error
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mae(self, y_true, y_pred):
        """
        Calculate the mean absolute error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            mean absolute error
        """
        return self.mean_absolute_error(y_true, y_pred)
    
    def mse(self, y_true, y_pred):
        """
        Calculate the mean squared error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            mean squared error
        """
        return self.mean_squared_error(y_true, y_pred)
    
    def rmse(self, y_true, y_pred):
        """
        Calculate the root mean squared error
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            root mean squared error
        """
        return self.root_mean_squared_error(y_true, y_pred)
    
    def r2(self, y_true, y_pred):
        """
        Calculate the R-squared value
        Parameters:
            y_true: true values
            y_pred: predicted values
        Returns:
            R-squared value
        """
        return self.r_squared(y_true, y_pred)

    def simple_linear_regression(self, x, y):
        """
        Simple Linear Regression
        Parameters:
            x: x values
            y: y values
        Returns:
            w: weights [slope, intercept]
        """
        # Add a bias term to X for the intercept
        X = np.vstack((x, np.ones(len(x)))).T
        # Solve for weights w = [slope, intercept] using the normal equation
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w