# An Introduction to Performant Python

This repository aims to serve as a tutorial style introduction to discovering numeric performance optimizations in the Python programming language. It assumes only a cursory knowledge of Python and a tolerance for pinches of math.

## Table of Contents:
1. [Preamble](#preamble)
2. [NumPy](#numpy)
    * [Duck Typing Limitations](#duck_limits)
    * [Broadcasting](#broadcasting)
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

As this repository intends simply to introduce NumPy we will be focusing on the first two bullet points, N-dimensional arrays and broadcasting functions. We should, however, take a moment to talk about some of the intuitions for when and why we can expect performance gains from NumPy.

<a name="duck_limits">
### Duck Typing Limitations
</a>

So we have some intuition for one of the reasons using NumPy might give us faster code let's discuss type systems for a moment.

Python is colloquially referred to a duck typed language, deriving from that phrase now known as the duck test, "If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck." You might be wondering, how in the world is this saying applicable to programming, aka, what's all this quack about? So let's get into that.

In Python, since it's an interpreted language, when you type the command `4 + 5` it doesn't __*know*__ what 4 and 5 are so it has to go check to see if you can add them together. As it turns out, you can, so Python with return `9`.

This going and checking if you can perform the requested operation in real time is part of what makes Python easy to program in. For example, if you want to do something in a `while` loop until a list, `ex_list`, is empty you could write `while len(ex_list) >= 0`. However, we can use duck typing to our advantage and instead write `while ex_list`. This works because Python knows `while` needs a boolean, so it will silently try to interpret `ex_list` as one; luckily a list only evaluates to `False` when it is empty, which is exactly what we want!

There is, however, a downside to duck typing. It incurs a time penalty, read: it can be slow sometimes. One of those times is when you're looping over numbers, and performing numeric operations on them. Even if you know you have a list only containing numbers, if you want to add 2 to every one of those numbers, you'll need to loop over the list and add 2 to every number; each time though, Python will need to manually check if it can add 2 to that element of the list. :'(

While much of the slowness in the above add 2 to all the elements of a list example can be attributed to duck typing pitfalls, we can see another aspect of the situation that is limiting is the nature of a Python list. Since Python lists are heterogeneous, this manual check for ability to add is actually necessary. Is there a different data structure, then, that could alleviate this pain point? I think you can guess where we're going.

<a name="broadcasting">
### Broadcasting
</a>

Before waxing poetic or getting incredibly abstract, bordering esoteric, about NumPy let us introduce an algorithm, k-means, as a medium to observe NumPy's power.

<a name="kmeans">
## k-means
</a>

[k-means](https://en.wikipedia.org/wiki/K-means_clustering) is an unsupervised machine learning algorithm designed to discover clusters within a dataset. Consider the following, highly math-less description, of k-means; algorithm/math/English junkies, please forgive me.

    Try to find the centers of blobs of that already "exist" within a dataset.

The following image, hopefully in the more intuitive visual way, demonstrates what is meant by "centers" and "blobs".

<div style="text-align: center"><img src="images/example_clustering.png" style="width: 300px"></div>

**Note:** In practice these "blobs" exist in a space with many more than two dimensions. The above plot is presented solely as a device to gain intuition for what we're trying to accomplish with k-means.

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

This k-means implementation lives under the name `base_python` in the `kmeans.py` script. The meat of it, however, is implemented in two functions: `get_new_assignments` and `calculate_new_centroids`. Let's look at both now.

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
