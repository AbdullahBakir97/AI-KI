### `jupyter_tips.md` Content

# Jupyter Tips & Tricks

## Introduction

This document provides useful tips and tricks for maximizing productivity with Jupyter Notebook and JupyterLab.

## Table of Contents

- [Jupyter Tips \& Tricks](#jupyter-tips--tricks)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
  - [Magic Commands](#magic-commands)
  - [Customizing Jupyter](#customizing-jupyter)
  - [Enhancing Notebooks](#enhancing-notebooks)
  - [Sharing Notebooks](#sharing-notebooks)
  - [Additional Resources](#additional-resources)
  - [Conclusion](#conclusion)

## Keyboard Shortcuts

- **Command Mode (Press `Esc` to activate):**
  - `H`: Show keyboard shortcuts
  - `A`: Insert cell above
  - `B`: Insert cell below
  - `M`: Change cell to Markdown
  - `Y`: Change cell to code
  - `D D`: Delete selected cell(s)

- **Edit Mode (Press `Enter` to activate):**
  - `Ctrl + Enter`: Run cell
  - `Shift + Enter`: Run cell and select below
  - `Alt + Enter`: Run cell and insert new cell below

## Magic Commands

- Line magic (`%`):
  - `%timeit`: Measure execution time of Python code
  - `%matplotlib inline`: Display matplotlib plots inline

- Cell magic (`%%`):
  - `%%time`: Measure execution time of entire cell
  - `%%html`: Render cell as HTML

## Customizing Jupyter

- **Jupyter Notebook Themes:**

  - Install `jupyterthemes` to apply custom themes:

    ```bash
    pip install jupyterthemes
    jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T
    ```

- **JupyterLab Extensions:**
  - Install and manage extensions using `jupyter labextension install`.

## Enhancing Notebooks

- **Widgets with ipywidgets:**
  - Create interactive widgets within Jupyter notebooks.

- **Markdown Extensions:**
  - Use markdown extensions for table of contents, LaTeX equations, and more.

## Sharing Notebooks

- **nbconvert:**
  - Convert Jupyter notebooks to other formats (HTML, PDF, etc.):
    ```bash
    jupyter nbconvert --to html notebook.ipynb
    ```

- **nbviewer:**
  - Share static Jupyter notebooks online using [nbviewer](https://nbviewer.jupyter.org/).

## Additional Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Jupyter Notebook Cheat Sheet](https://www.cheatography.com/weidadeyue/cheat-sheets/jupyter-notebook/pdf/)
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/)

## Conclusion

These tips and tricks will help you leverage Jupyter Notebook and JupyterLab's capabilities, enhancing your productivity and collaboration in data science and computational workflows.
