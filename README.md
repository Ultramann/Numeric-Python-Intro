# An Introduction to Performant Python

This repository aims to serve as a tutorial style introduction to discovering numeric performance optimizations in the Python programming language. It assumes only a cursory knowledge of Python and a tolerance for pinches of math.

## Table of Contents:
1. [Preamble](#preamble)
2. [NumPy](#numpy)
    * [Duck Typing vs. Static Typing](#duck_v_static)
3. [k-means](#kmeans)
    * [Why k-means?](#whyk)
    * [Implementations](#implementations)
        1. [Base Python](#base_python)
        2. [NumPy Accelerated](#np_accel)
4. [Postmortem](#postmortem)

<a name="preamble">
## Preamble
</a>

Python is well known to be an incredible learning language for beginning programmers on account of its readability, intuitive syntax, and wonderful community.

What many people don't know about Python when they unknowingly choose it from a stack of languages based on what programming language the internet suggests a beginner start with is that Python has a vast array (no pun intended) of other amazing features, libraries, and applications; many more in fact than a beginner could even begin to guess.

A short, but far from exhaustive, list of fantastic Python offerings to whet your technical palate:
* [Django](https://www.djangoproject.com/) - web development.
* [Selenium-Python](http://selenium-python.readthedocs.io/) - web browser automation.
* [SciPy](https://www.scipy.org/) - scientific computing.
* [Scikit-Learn](http://scikit-learn.org/stable/) - machine learning.
* [Tensorflow](https://www.tensorflow.org/) - nuff said.

It is through a specific library within the SciPy ecosystem that this repository will introduce programming performant Python. That library is [NumPy](http://www.numpy.org/).

<a name="numpy">
## NumPy 
</a>

NumPy's front page defines itself as the following:

> NumPy is the fundamental package for scientific computing with Python. It contains among other things:
> * a powerful N-dimensional array object
> * sophisticated (broadcasting) functions
> * tools for integrating C/C++ and Fortran code
> * useful linear algebra, Fourier transform, and random number capabilities

These are amazing features indeed if we're looking for a library to add some performance to our Python code.

As this repository intends simply to introduce NumPy we will be focusing on the first two bullet points, N-dimensional arrays and broadcasting functions. We should, however, take a moment to talk about some of the intutions for when and why we can expect performance gains from NumPy.

<a name="duck_v_static">
### Duck Typing vs. Static Typing
</a>

**Talk about pythons duck typing against numpy's knowing what is in the array.**

Before waxing poetic or getting incredibly abstract, bordering esoteric, about NumPy let us introduce an algorithm, k-means, as a medium to observe NumPy's power.

<a name="kmeans">
## k-means
</a>

[k-means](https://en.wikipedia.org/wiki/K-means_clustering) is an unsupervised machine learning algorithm designed to discover clusters within a dataset. Consider the following, highly math-less description, of k-means (algorithm/math/English junkies, please forgive me):

    Try to find the centers of blobs of that already "exist" within a dataset.

The following image, hopefully in the more intuitive visual way, demonstrates what is meant by "centers" and "blobs".

![Example Clustering](images/example_clustering.png)

**Note:** In practice these "blobs" exist in much high dimensions than two. The above plot is presented solely as a devise to gain intuition for what we're trying to accomplish with k-means.

<a name="whyk">
### Why k-means?
</a>

**Talk about why k-means is a good algorithm to choose to demonstrate performant python.**

In order to stay on the rails and keep chugging towards learning about injecting performance into Python and not detour deep into machine learning algorithms...ville, let's move on to implementations of the k-means algorithm.

<a name="implementations">
### Implementations
</a>


**Note:** The code blocks throughout the remainder of this tutorial will not include doc strings to cut down on unnecessary space usage. In addition, code blocks will be accompanied by descriptions of the code. Also! Doc strings are important! As such, they are obviously included in the actual scripts.

First we're going to to look at an implementation using only built-in python functions and data types.

<a name="base_python">
#### Base Python
</a>

This k-means implementation lives under the name `base_python` in the `kmeans.py` script. The meat of it however is implemented in two functions: `get_new_assignments` and `calcualte_new_centroids`. Let's look at both now.

##### Getting New Centroid Assignments
```python
def get_new_assignments(centroids, X):
    centroid_assignments = [[] for _ in centroids]
    for x in X:
        closest_dist = 1e100
        closest_centroid = None
        for centroid_idx, centroid_location in enumerate(centroids):
            current_dist = list_euclidean_dist(centroid_location, x)
            if current_dist < closest_dist:
                closest_dist = current_dist
                closest_centroid = centroid_idx
        centroid_assignments[closest_centroid].append(x)
    return centroid_assignments
```

<a name="np_accel">
### NumPy Accelerated
</a>

**Talk about the numpy version of the algorithm**

<a name="postmortem">
## Postmortem
</a>

**Talk about the high level intuition of why nuumpy sped things up for kmeans (c level speed by removing duck checks) and begin to talk about other optimization that we can get from numpy (linear algebra stuff, calling down to c and fortran, revisit numpy's self description**
