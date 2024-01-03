import numpy as np
import matplotlib.pyplot as plt

# Define the constant H0 value, for example, let's take H0 = 1 for illustration
H0_value = 0.25

# Define the range for X and Y
X_values = np.linspace(0, 1, 400)
Y_values = np.linspace(0, 1, 400)

# Since Y is independent and does not affect P, we only need to calculate P as a function of X
P_values = 6 * X_values * (1 - X_values) / ((H0_value + 1 - X_values)**2 * (1 + 2 * H0_value))

# Plotting the 2D plot
plt.figure(figsize=(8, 6))
plt.plot(X_values, P_values, label='P(X)')
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Plot of P as a function of X with H0={}'.format(H0_value))
plt.legend()
plt.show()

# Creating a meshgrid for X and Y to calculate P for each (X,Y) pair
# Since Y does not affect the value of P, P will be the same for all values of Y for a given X.
X_grid, Y_grid = np.meshgrid(X_values, Y_values)
P_grid = 6 * X_grid * (1 - X_grid) / ((H0_value + 1 - X_grid)**2 * (1 + 2 * H0_value))

# Plotting the 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_grid, Y_grid, P_grid, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('P(X,Y)')
ax.set_title('Surface Plot of P(X,Y) with H0={}'.format(H0_value))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
