# PECAN: Persistent Condensation

[![arXiv](https://img.shields.io/badge/arXiv-2203.14860-b31b1b.svg)](https://arxiv.org/abs/2203.14860) ![GitHub contributors](https://img.shields.io/github/contributors/KrishnaswamyLab/PECAN) ![GitHub](https://img.shields.io/github/license/KrishnaswamyLab/PECAN)

This repository contains the code for our work on [*Time-Inhomogeneous
Diffusion Geometry and
Topology*](https://epubs.siam.org/doi/10.1137/21M1462945). If you use
this code, please consider citing our paper:

```biblatex
@article{Huguet23a,
  title         = {Time-Inhomogeneous Diffusion Geometry and Topology},
  author        = {Huguet, Guillaume and Tong, Alexander and Rieck, Bastian and Huang, Jessie and Kuchroo, Manik and Hirn, Matthew and Wolf, Guy and Krishnaswamy, Smita},
  journal       = {SIAM Journal on Mathematics of Data Science},
  year          = 2023,
  volume        = 5,
  number        = 2,
  pages         = {346--372},
  doi           = {10.1137/21M1462945},
  primaryclass  = {cs.LG},
  archiveprefix = {arXiv},
  eprint        = {2203.14860},
  abstract      = {Abstract. Diffusion condensation is a dynamic process that yields a sequence of multiscale data representations that aim to encode meaningful abstractions. It has proven effective for manifold learning, denoising, clustering, and visualization of high-dimensional data. Diffusion condensation is constructed as a time-inhomogeneous process where each step first computes a diffusion operator and then applies it to the data. We theoretically analyze the convergence and evolution of this process from geometric, spectral, and topological perspectives. From a geometric perspective, we obtain convergence bounds based on the smallest transition probability and the radius of the data, whereas from a spectral perspective, our bounds are based on the eigenspectrum of the diffusion kernel. Our spectral results are of particular interest since most of the literature on data diffusion is focused on homogeneous processes. From a topological perspective, we show that diffusion condensation generalizes centroid-based hierarchical clustering. We use this perspective to obtain a bound based on the number of data points, independent of their location. To understand the evolution of the data geometry beyond convergence, we use topological data analysis. We show that the condensation process itself defines an intrinsic condensation homology. We use this intrinsic topology, as well as the ambient persistent homology, of the condensation process to study how the data changes over diffusion time. We demonstrate both types of topological information in well-understood toy examples. Our work gives theoretical insight into the convergence of diffusion condensation and shows that it provides a link between topological and geometric data analysis.}
}
```

## Installing the package

The recommended way of installing the package entails using
[`poetry`](https://python-poetry.org/) for Python. After cloning this
repository, everything can be set up using the following command, with
`$` indicating that the command is to be executed in your terminal:

```
$ poetry install 
```

## Using the package

The simplest way of using the package uses a shell spawned by `poetry`:

```
$ poetry shell
```

You should now be able to call the main scripts of the package.

## Running the algorithm

To run the algorithm on a 'double annulus' data set, use the following
command:

```
$ python condensation.py -n 256 -d double_annulus -o double_annulus.npz
```

This will create a file `double_annulus.npz` containing information
about the condensation process. By changing the parameter `-n`, you
can control the number of points that are to be used.

## Visualising the results

To visualise the results, you can use either
`visualise_diffusion_homology.py` or `visualise_persistent_homology.py`,
depending on what type of features you are interested in. Both of these
scripts accept a file generated by `condensation.py` as an input. For
instance, to show the diffusion homology of the 'double annulus' data
set over time, use the following command:

```
$ python visualise_diffusion_homology.py double_annulus.npz
```

Note that *all* visualisations are meant for research purposes only and
are subject to change as the project progresses.

## Using the `Rivet` callback

The code now supports *bifiltrations*. To make use of this
functionality, you need to install [`rivet_console`](https://github.com/rivetTDA/rivet).
If you are on Mac OS X, you can use [HomeBrew](https://brew.sh) for this
purpose:

```
$ brew tap BorgwardtLab/mlcb
$ brew install rivet
```

Regardless of the installation method, make sure that you tell `poetry`
that additional dependencies are present:

```
$ poetry update
```

This ensures that the Python 'wrapper' for `rivet_console` is installed.

## Using `Oineus` bindings

[`Oineus`](https://github.com/anigmetov/oineus) is a fast C++ library for
calculating persistent homology. It comes with Python bindings, which
we have wrapped in [`py-oineus`, an **experimental package**](https://github.com/aidos-lab/py-oineus).
The module requires a recent C++ compiler (supporting C++17) as well as
support for [Threading Building Blocks](https://github.com/oneapi-src/oneTBB).
If you install those library, for instance via `brew install tbb`, you
can add the `oineus` Python bindings to the project like this:

```
$ poetry shell
$ pip install git+ssh://git@github.com/aidos-lab/py-oineus
```

To test the integration, run `python pecan/oineus_integration.py` in the
proper virtual environment.
