### `matplotlib_basics.md` Content


# Matplotlib Basics

## Introduction

Matplotlib is a popular plotting library for Python that provides a MATLAB-like interface for creating a variety of plots and visualizations. This guide covers the basics of using Matplotlib to create simple plots, customize visuals, and visualize data effectively.

## Table of Contents

- [Matplotlib Basics](#matplotlib-basics)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installing Matplotlib](#installing-matplotlib)
  - [Basic Plotting with Matplotlib](#basic-plotting-with-matplotlib)
    - [Line Plot](#line-plot)
    - [Scatter Plot](#scatter-plot)
  - [Customizing Plots](#customizing-plots)
    - [Adding Annotations](#adding-annotations)
    - [Changing Styles and Colors](#changing-styles-and-colors)
  - [Types of Plots](#types-of-plots)
    - [Histogram](#histogram)
    - [Bar Chart](#bar-chart)
  - [Saving and Exporting Plots](#saving-and-exporting-plots)
    - [Saving Plots](#saving-plots)
  - [Additional Resources](#additional-resources)
  - [Conclusion](#conclusion)

---

## Installing Matplotlib

To install Matplotlib, refer to the [Matplotlib Installation Guide](../installation/matplotlib_installation.md).

---

## Basic Plotting with Matplotlib

### Line Plot

Create a basic line plot using Matplotlib:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.grid(True)
plt.show()
```

### Scatter Plot

Generate a scatter plot with random data:

```python
x = np.random.rand(100)
y = np.random.rand(100)
sizes = np.random.rand(100) * 100

plt.scatter(x, y, s=sizes, alpha=0.5)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Sizes')
plt.grid(True)
plt.show()
```

---

## Customizing Plots

### Adding Annotations

Annotate points or lines on plots for better understanding:

```python
plt.plot(x, y)
plt.annotate('Maximum', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

### Changing Styles and Colors

Customize line styles, markers, and colors:

```python
plt.plot(x, y, linestyle='--', marker='o', color='r')
plt.show()
```

---

## Types of Plots

### Histogram

Create a histogram to visualize data distribution:

```python
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Random Data')
plt.grid(True)
plt.show()
```

### Bar Chart

Generate a bar chart for categorical data:

```python
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

plt.bar(categories, values)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Chart')
plt.grid(True)
plt.show()
```

---

## Saving and Exporting Plots

### Saving Plots

Save plots in various formats such as PNG, PDF, or SVG:

```python
plt.savefig('plot.png')
```

---

## Additional Resources

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

## Conclusion

Matplotlib's versatility and ease of use make it an indispensable tool for data visualization in Python. Explore its features and capabilities to create impactful visualizations for your data analysis projects.

