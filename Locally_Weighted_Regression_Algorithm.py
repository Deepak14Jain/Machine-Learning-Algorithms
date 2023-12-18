import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_regression(X, y, x_test, tau=0.1):
    m = len(X)
    x_test = np.array([1, x_test])
    weights = np.exp(-((X - x_test[1]) * 2) / (4 * tau))
    W = np.diag(weights)
    X = np.column_stack((np.ones(m), X))
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    y_pred = x_test @ theta
    return y_pred


data = pd.read_csv("p11dataset.csv")
X = np.array([1,3,5,7,9])
y = np.array([2.5, 6.7, 8.9, 9.0, 12.8])
new_x = float(input("Enter a new value for X: "))
predicted_y = locally_weighted_regression(X, y, new_x)
print(f"Predicted Y for X={new_x}: {predicted_y:.4f}")

plt.scatter(X, y, label='Data')
plt.scatter(new_x, predicted_y, color='red', label='Predicted Point')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
