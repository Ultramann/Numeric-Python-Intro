# Numeric Python Intro

## NumPy Demonstration with k-means Algorithm

The code in this directory forms the basis of a introductory tutorial on numeric performance optimizations for the k-means clustering algorithm in the Python programming language that I put on for the San Francisco Python Meetup Group. Discussion of the code and approach can be found in my [blog](https://ultramann.github.io/posts/numeric_python/).

Python libraries dependencies: `scipy`, `numpy`, and `matplotlib`.

## Notes

The code in this directory is for teaching purposes. As such, many of the choices made in writing it were towards the end of more readable/explainable code. In addition, as the topic being taught in this repository is numeric python, the focus is not on the most cutting edge of k-means techniques. Nor does it embrace Python's object oriented paradigm which would likely prefer that the algorithm be implemented as a class.

The `demo.py` script is equipped with flags to perform different demonstrations. Help for this function can be found by running `python demo.py -h` at the command line. Example outputs with various settings can be found in the `demo_examples.ipynb` notebook.
