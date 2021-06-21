# PECAN: Persistent Condensation

This repository contains our work-in-progress code for
a persistence-enhanced variant of the [diffusion condensation
algorithm](https://ieeexplore.ieee.org/document/9006013).

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
