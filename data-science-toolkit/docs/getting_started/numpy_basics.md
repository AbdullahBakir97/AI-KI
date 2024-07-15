### `numpy_basics.md` Content


# NumPy Basics

## Introduction

NumPy (Numerical Python) is a fundamental library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. This guide covers the basics of NumPy, its key features, and how to use it effectively for data manipulation and computation.

## Table of Contents

- [NumPy Basics](#numpy-basics)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installing NumPy](#installing-numpy)
  - [NumPy Arrays](#numpy-arrays)
    - [Creating NumPy Arrays](#creating-numpy-arrays)
    - [Array Attributes](#array-attributes)
  - [Basic Array Operations](#basic-array-operations)
    - [Element-wise Operations](#element-wise-operations)
    - [Array Manipulation](#array-manipulation)
  - [Universal Functions (ufuncs)](#universal-functions-ufuncs)
    - [Mathematical Functions](#mathematical-functions)
  - [Indexing and Slicing](#indexing-and-slicing)
    - [Indexing](#indexing)
    - [Slicing](#slicing)
  - [Broadcasting](#broadcasting)
  - [Linear Algebra Operations](#linear-algebra-operations)
    - [Matrix Operations](#matrix-operations)
  - [Additional Resources](#additional-resources)
  - [Conclusion](#conclusion)

---

## Installing NumPy

To install NumPy, refer to the [NumPy Installation Guide](../installation/numpy_installation.md).

---

## NumPy Arrays

### Creating NumPy Arrays

Create NumPy arrays from Python lists or using built-in functions:

```python
import numpy as np

# Create a 1D array
arr1 = np.array([1, 2, 3, 4, 5])

# Create a 2D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
```

### Array Attributes

Explore array properties such as shape, size, and data type:

```python
print(arr1.shape)   # (5,)
print(arr2.shape)   # (2, 3)
print(arr1.dtype)   # int64
```

---

## Basic Array Operations

### Element-wise Operations

Perform basic arithmetic operations on arrays:

```python
arr3 = arr1 + arr2
arr4 = arr1 * 2
```

### Array Manipulation

Reshape, concatenate, and split arrays for data manipulation:

```python
arr5 = arr2.reshape(3, 2)
arr6 = np.concatenate((arr1, arr2.flatten()))
```

---

## Universal Functions (ufuncs)

### Mathematical Functions

Apply mathematical functions element-wise on arrays:

```python
arr7 = np.sqrt(arr1)
arr8 = np.exp(arr2)
```

---

## Indexing and Slicing

### Indexing

Access and modify elements of an array using indexing:

```python
print(arr1[0])       # 1
arr1[0] = 10
print(arr1)          # [10, 2, 3, 4, 5]
```

### Slicing

Extract subarrays using slicing:

```python
print(arr2[:, 1])    # [2, 5]
print(arr2[1, :])    # [4, 5, 6]
```

---

## Broadcasting

Automatically perform operations on arrays with different shapes:

```python
arr9 = arr1 + 10
```

---

## Linear Algebra Operations

### Matrix Operations

Perform matrix multiplication and other linear algebra operations:

```python
arr10 = np.dot(arr2, arr2.T)
```

---

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)

---

## Conclusion

NumPy's powerful array operations and mathematical functions make it an essential tool for scientific computing and data analysis in Python. Mastering NumPy fundamentals enhances your ability to manipulate and analyze data efficiently.

