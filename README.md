# Neural Discovery of Permutation Subgroups

This repository contains the code associated with the paper **"A Unified Framework for Discovering Discrete Symmetries"** accepted as contributed talk (poster presentation) at [AISTATS 2024](http://aistats.org/aistats2024/).

### Short description
We consider the problem of learning a function respecting a symmetry from among a class of symmetries. We develop a unified framework that enables symmetry discovery across a broad range of subgroups including locally symmetric, dihedral and cyclic subgroups. At the core of the framework is a novel architecture composed of linear, matrix-valued and non-linear functions that expresses functions invariant to these subgroups in a principled manner. The structure of the architecture enables us to leverage multi-armed bandit algorithms and gradient descent to efficiently optimize over the linear and the non-linear functions, respectively, and to infer the symmetry that is ultimately learnt. We also discuss the necessity of the matrix-valued functions in the architecture. Experiments on image-digit sum and polynomial regression tasks demonstrate the effectiveness of our approach.
