# MouseFlowShape

Shape analysis of mouse embryo surfaces from light-sheet volumes, using
conformal sphere mapping and spherical-harmonic decomposition of curvature
([FlowShape](https://github.com/DeBoerLab/flowshape)).

> Work in progress. Paths in the scripts are currently hardcoded to local
> and network drives and will need editing before the pipeline runs elsewhere.

## Pipeline

**1. `Mesh_Generation/` (MATLAB)** — image volumes to watertight surface meshes.

| File | Role |
| --- | --- |
| `process_meshes_example.m` | Driver: reads KLB stacks, thresholds, extracts the surface point cloud, closes the embryonic cup with an alpha shape, shrink-wraps, Poisson-remeshes, writes one `.obj` per timepoint |
| `shrinkwrap.m` | Iterative shrink-wrap of a subdivided convex hull onto a point cloud |
| `upsample_triangulation.m` | Midpoint subdivision of a triangulation |

**2. `Spherical_Harmonics/` (Python)** — meshes to shape descriptors.

| File | Role |
| --- | --- |
| `run_flowshape_example.ipynb` | Driver notebook: heal → orient → mask → decompose → save → render |
| `mesh_functions.py` | Mesh healing (pymeshlab/pymeshfix), interactive rotation / mask-plane / view widgets, ICP alignment, frame rendering and movie stitching |
| `helper_functions.py` | Dataset loading, Gaussian frequency filters, spherical-harmonic decomposition (`IRF`), HDF5 save/load |

Per timepoint the notebook computes the sphere map, the curvature function
`rho`, a soft per-face mask from a user-placed cutting plane, and the
resulting `l_max = 16` harmonic coefficients (`weights_masked`), stored in
`results.h5`.

## Environment

Conda/pip commands used to build the working environment are recorded in
`flowshape_mymachine.txt`. MATLAB side requires the Image Processing,
Statistics & Machine Learning, Computer Vision and Lidar toolboxes, plus the
[keller-lab-block-filetype](https://github.com/KellerLabMPI/keller-lab-block-filetype)
MATLAB wrapper for reading `.klb` stacks.

## Data

Image volumes, meshes and results are not tracked here — see `.gitignore`.
