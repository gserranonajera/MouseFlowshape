import os
import glob

import numpy as np
import h5py
import igl

import flowshape
import scipy
import pickle
from tqdm import tqdm


def load_dataset(data_folder: str, ti: int = 0, tf: int = None, skip: int = 1) -> list[dict]:
    """Load OBJ meshes from a folder into a list of dicts with 'v' and 'f' keys.

    Parameters
    ----------
    data_folder : str
        Path to folder containing .obj files.
    ti, tf, skip : int
        Slice parameters applied to the sorted file list: files[ti:tf:skip].
    """
    files = sorted(glob.glob(os.path.join(data_folder, "*.obj")))
    files = files[ti:tf:skip]
    data = []
    for path in tqdm(files, desc="Loading OBJs"):
        v, f = igl.read_triangle_mesh(path)
        data.append({"v": v, "f": f})
    return data

# helper functions
l_max = 16 # number of harmonics, normally 16 -> 16^2

# gaussian filter. blurs higher frequencies.
# k controls how fast is the decay of the contribution of higher frequencies
def build_gaussian(k=0.01, l_max=16):
    ww = np.zeros(l_max**2)
    for l in range(0, l_max):
        i = int(l) ** 2
        i2 = int(l + 1) ** 2

        ww[i:i2] = np.exp(-k * l * (l + 1))
    return ww

# laplacian of gaussian filter
# edge detection:  highlights regions where curvature changes at a characteristic spatial scale controlled by k
# currently not in use
def build_filter(k=0.01, l_max=16):
    ww = np.zeros(l_max**2)
    for l in range(0, l_max):
        i = int(l) ** 2
        i2 = int(l + 1) ** 2
        ww[i:i2] = np.exp(1 - k * l * (l + 1)) * k * l * (l + 1)
    return ww

# decomposes a scalar function rho (defined per-face on the mesh) into spherical harmonic coefficients. 
def irf(v_sphere, f, rho):
    v_bary = igl.barycenter(v_sphere, f) # computes the centroid of each triangle on the sphere map
    v_bary = flowshape.project_sphere(v_bary) # projects those barycenters back onto the unit sphere
    W = 0.5 * igl.doublearea(v_sphere, f) # computes the area of each triangle on the sphere
    W = scipy.sparse.diags(W) #  converts to a diagonal matrix for efficient matrix multiplication
    weights, Y_mat = flowshape.IRF_scalar(rho, v_bary, W, max_degree=l_max) #  spherical harmonic decomposition

    # returns:
    # weights (the 256 spherical harmonic coefficients)
    # Y_mat (the basis matrix, where each column is a spherical harmonic evaluated at all barycenters)
    return weights, Y_mat #rho ≈ Y_mat @ weights


def compute_harmonics_single(
    v: np.ndarray,
    f: np.ndarray,
    f_mask: np.ndarray,
    smoothing_mask: np.ndarray,
) -> dict:
    """Compute spherical harmonic descriptors for a single mesh.

    Returns a dict with keys: v, f, f_mask, rho, weights_mask,
    weights_masked, Y_mat, vs, computed (1 on success, 0 on failure).
    """
    try:
        c, _ = igl.orientable_patches(f)
        f, _ = igl.orient_outward(v, f, c)

        v_sphere = flowshape.sphere_map(v, f)
        rho = flowshape.curvature_function(v, v_sphere, f)

        weights_mask, Y_mat = irf(v_sphere, f, f_mask)
        mask_smooth = Y_mat @ (weights_mask * smoothing_mask)
        weights_masked, Y_mat = irf(v_sphere, f, rho * mask_smooth)

        return {
            "v": v,
            "f": f,
            "f_mask": f_mask,
            "rho": rho,
            "weights_mask": weights_mask,
            "weights_masked": weights_masked,
            "Y_mat": Y_mat,
            "vs": v_sphere,
            "computed": 1,
        }
    except Exception:
        return {
            "v": v,
            "f": f,
            "f_mask": f_mask,
            "computed": 0,
        }


def save_results(
    filepath: str,
    time_points: list[dict],
    params: dict | None = None,
):
    """Save time_points to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output .h5 path.
    time_points : list[dict]
        Each dict may contain: v, f, f_mask, rho, vs, Y_mat,
        weights_mask, weights_masked, computed.
    params : dict or None
        Transform parameters (T_rotation, mask_plane_normal, etc.)
        stored under the 'params' group.
    """
    with h5py.File(filepath, "w") as hf:
        if params is not None:
            g = hf.create_group("params")
            for key, val in params.items():
                if isinstance(val, str):
                    g.attrs[key] = val
                else:
                    g.create_dataset(key, data=np.array(val))

        for i, tp in enumerate(tqdm(time_points, desc="Saving HDF5")):
            g = hf.create_group(f"{i:04d}")
            for key, val in tp.items():
                g.create_dataset(key, data=np.array(val))


def load_results(filepath: str, keys: list[str] | None = None) -> list[dict]:
    """Load time_points from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Input .h5 path.
    keys : list[str] or None
        If provided, only load these keys per time point
        (e.g. ['weights_masked', 'computed'] for PCA).
        If None, load everything.

    Returns
    -------
    list[dict], one per time point.
    """
    time_points = []
    with h5py.File(filepath, "r") as hf:
        groups = sorted(k for k in hf.keys() if k != "params")
        for name in tqdm(groups, desc="Loading HDF5"):
            g = hf[name]
            tp = {}
            load_keys = keys if keys is not None else list(g.keys())
            for key in load_keys:
                if key in g:
                    val = g[key][()]
                    if val.ndim == 0:
                        val = val.item()
                    tp[key] = val
            time_points.append(tp)
    return time_points


def load_params(filepath: str) -> dict:
    """Load just the params group from an HDF5 results file."""
    params = {}
    with h5py.File(filepath, "r") as hf:
        if "params" in hf:
            g = hf["params"]
            for key in g:
                val = g[key][()]
                if val.ndim == 0:
                    val = val.item()
                params[key] = val
            for key, val in g.attrs.items():
                params[key] = val
    return params