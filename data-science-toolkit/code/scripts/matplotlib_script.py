"""
Matplotlib Script Example

This script demonstrates how to create various types of plots using Matplotlib,
customize plot appearance, and save plots to files.


"""

import matplotlib.pyplot as plt
import numpy as np

# Example data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and axis (subplot)
fig, ax = plt.subplots()

# Plot data on the first axis
ax.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)
ax.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2, marker='x', markersize=5)

# Customize plot appearance
ax.set_title('Sine and Cosine Functions')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()

# Add gridlines
ax.grid(True)

# Save the plot as a PNG file
plt.savefig('sine_cosine_plot.png')

# Display the plot
plt.show()



