import numpy as np
import matplotlib.pyplot as plt

# True function definition
true_function = lambda x1, x2: x1**2 + 2*x2 + 3*x1*x2 + 4

# Model definition
def model(x1, x2, q1, q2, q3):
    return q1 * x1 + q2 * x2 + q3

# Gradient descent function with loss tracking
def gradient_descent(x1, x2, y, learning_rate=0.01, epochs=100):
    q1, q2, q3 = np.random.rand(3)  # Initialize parameters
    m = len(y)
    losses = []  # Initialize a list to store losses

    for epoch in range(epochs):
        y_pred = model(x1, x2, q1, q2, q3)
        mse = np.mean((y - y_pred)**2)  # Calculate MSE
        losses.append(mse)  # Track loss over iterations

        dq1 = (-2/m) * np.dot(x1, (y - y_pred))
        dq2 = (-2/m) * np.dot(x2, (y - y_pred))
        dq3 = (-2/m) * np.sum(y - y_pred)

        # Update parameters
        q1 -= learning_rate * dq1
        q2 -= learning_rate * dq2
        q3 -= learning_rate * dq3

    return q1, q2, q3, losses  # Return losses

# Create and train on multiple datasets
num_datasets = 2
results = []

# For tracking overall MSEs across datasets
all_losses = []

for i in range(num_datasets):
    np.random.seed(i)  # Seed for reproducibility
    x1, x2 = np.random.rand(2, 100)
    y = true_function(x1, x2)
    q1, q2, q3, losses = gradient_descent(x1, x2, y)
    y_pred = model(x1, x2, q1, q2, q3)
    results.append((x1, x2, y, y_pred, q1, q2, q3))
    all_losses.append(losses)  # Store losses for each dataset

# Visualization of true vs predicted values for each dataset
plt.figure(figsize=(10, 10))

for i, (x1, x2, y, y_pred, q1, q2, q3) in enumerate(results):
    plt.subplot(1, num_datasets, i + 1)  # Adjust for the number of datasets
    plt.scatter(y, y_pred, alpha=0.7, color='blue', label='Predicted')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Ideal Prediction')  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Dataset {i + 1}: q1={q1:.2f}, q2={q2:.2f}, q3={q3:.2f}')
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
# Plot the MSE over time (for each dataset)
for i, losses in enumerate(all_losses):

    plt.subplot(1, num_datasets, i + 1)  # Adjust for the number of datasets
    plt.plot(losses, color="green")
    plt.title(f"Mean Squared Error (MSE) Over Iterations for Dataset {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")

plt.show()
# Final parameters for each dataset
for i, (q1, q2, q3) in enumerate([(res[4], res[5], res[6]) for res in results]):
    print(f"Final parameters for Dataset {i + 1}: q1 = {q1:.2f}, q2 = {q2:.2f}, q3 = {q3:.2f}")
