# MouseFlowShape

Shape analysis of mouse embryo surfaces from light-sheet volumes, using
conformal sphere mapping and spherical-harmonic decomposition of curvature
([FlowShape](https://github.com/DeBoerLab/flowshape)).

> Paths in the scripts are currently hardcoded to local
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

### Python

```bash
conda create -n flowshape
conda activate flowshape
conda install python=3.12
conda install -c conda-forge gdk-pixbuf=2.42.12 --force-reinstall
conda install -c conda-forge numpy scipy==1.14.1 jupyterlab scikit-learn scikit-image matplotlib pyvista meshplot
pip install pyacvd pymeshfix libigl==2.5.1 plyfile
pip install flowshape --no-deps
pip install tqdm
pip install pyklb
```

Then, for HDF5 I/O, mesh healing, parallel loops, ICP alignment and the
notebook kernel:

```bash
pip install h5py pymeshlab joblib open3d ipykernel
pip install numpy==1.26.0 tifffile==2024.12.12 imageio==2.37.0 imageio-ffmpeg==0.6.0 --force-reinstall
python -m ipykernel install --user --name flowshape --display-name "Flowshape"
```

Note the `numpy==1.26.0` downgrade at the end — several of the mesh
packages are not yet built against NumPy 2.x, so this pin is deliberate and
must come last. `flowshape` is installed with `--no-deps` for the same
reason.

### MATLAB

MATLAB side requires the Image Processing,
Statistics & Machine Learning, Computer Vision and Lidar toolboxes, plus the
[keller-lab-block-filetype](https://github.com/KellerLabMPI/keller-lab-block-filetype)
MATLAB wrapper for reading `.klb` stacks.
