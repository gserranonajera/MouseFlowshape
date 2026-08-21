import os
import numpy as np
import meshplot
import igl
import pymeshfix
import pymeshlab
import ipywidgets as widgets
from scipy.spatial.transform import Rotation
from IPython.display import display
import json
import tqdm
import open3d as o3d

def interactive_rotation_widget(v: np.ndarray, f: np.ndarray):
    """Display an interactive 3D mesh viewer with rotation/flip controls.

    Returns a mutable dict with a 'T_rotation' key,
    updated when the user clicks "Save T_rotation".
    """
    centroid = v.mean(axis=0)
    v_c = v - centroid
    scale = np.abs(v_c).max()
    v_c = v_c / scale

    p = meshplot.plot(v_c, f, return_plot=True, shading={"width": 800, "height": 600})

    p._renderer.controls[0].enableRotate = True
    p._renderer.controls[0].enableZoom = True
    p._renderer.controls[0].enablePan = True
    p._renderer.camera.position = [0, 0, 3]
    p._renderer.camera.up = [0, 1, 0]
    p._renderer.controls[0].target = [0, 0, 0]

    axis_origin = np.array([-1.5, -1.5, -1.5])
    axis_length = 0.8
    for i, col in enumerate(["#ff0000", "#00ff00", "#0000ff"]):
        tip = axis_origin.copy()
        tip[i] += axis_length
        p.add_lines(axis_origin, tip, shading={"line_color": col})

    rot_x = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot X', continuous_update=True)
    rot_y = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot Y', continuous_update=True)
    rot_z = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot Z', continuous_update=True)
    flip_x = widgets.Checkbox(value=False, description='Flip X')
    flip_y = widgets.Checkbox(value=False, description='Flip Y')
    flip_z = widgets.Checkbox(value=False, description='Flip Z')
    out = widgets.Output()

    state = {"T_baked": np.eye(3), "T_rotation": np.eye(3)}

    def get_current_T():
        R = Rotation.from_euler('xyz', [rot_x.value, rot_y.value, rot_z.value], degrees=True).as_matrix()
        S = np.diag([
            -1.0 if flip_x.value else 1.0,
            -1.0 if flip_y.value else 1.0,
            -1.0 if flip_z.value else 1.0,
        ])
        return S @ R

    def update(*args):
        T_total = get_current_T() @ state["T_baked"]
        p.update_object(vertices=v_c @ T_total.T)
        with out:
            out.clear_output()
            print(f"T_rotation = np.array({np.round(T_total, 6).tolist()})")

    def bake(b):
        state["T_baked"] = get_current_T() @ state["T_baked"]
        for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
            w.unobserve(update, names='value')
        rot_x.value = rot_y.value = rot_z.value = 0
        flip_x.value = flip_y.value = flip_z.value = False
        for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
            w.observe(update, names='value')
        with out:
            out.clear_output()
            print("Baked!")
            print(f"T_rotation = np.array({np.round(state['T_baked'], 6).tolist()})")

    def reset_all(b):
        state["T_baked"] = np.eye(3)
        bake(b)

    btn_bake = widgets.Button(description="Bake rotation", button_style='success')
    btn_bake.on_click(bake)
    btn_reset = widgets.Button(description="Reset all", button_style='danger')
    btn_reset.on_click(reset_all)

    for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
        w.observe(update, names='value')

    btn_save = widgets.Button(description="Save T_rotation", button_style='info')

    def save_T(b):
        state["T_rotation"] = get_current_T() @ state["T_baked"]
        with out:
            out.clear_output()
            print("Saved to T_rotation!")
            print(f"T_rotation = np.array({np.round(state['T_rotation'], 6).tolist()})")

    btn_save.on_click(save_T)

    btn_xy = widgets.Button(description="XY view")
    btn_xz = widgets.Button(description="XZ view")
    btn_yz = widgets.Button(description="YZ view")

    def set_view_xy(b):
        p._renderer.camera.position = [0, 0, 3]
        p._renderer.camera.up = [0, 1, 0]
        p._renderer.controls[0].target = [0, 0, 0]

    def set_view_xz(b):
        p._renderer.camera.position = [0, 3, 0]
        p._renderer.camera.up = [0, 0, 1]
        p._renderer.controls[0].target = [0, 0, 0]

    def set_view_yz(b):
        p._renderer.camera.position = [3, 0, 0]
        p._renderer.camera.up = [0, 1, 0]
        p._renderer.controls[0].target = [0, 0, 0]

    btn_xy.on_click(set_view_xy)
    btn_xz.on_click(set_view_xz)
    btn_yz.on_click(set_view_yz)

    display(widgets.VBox([
        widgets.HBox([rot_x, rot_y, rot_z]),
        widgets.HBox([flip_x, flip_y, flip_z]),
        widgets.HBox([btn_bake, btn_reset, btn_save]),
        widgets.HBox([btn_xy, btn_xz, btn_yz]),
        out
    ]))

    return state


def interactive_mask_plane_widget(
    v: np.ndarray,
    f: np.ndarray,
    T_rotation: np.ndarray,
    results_folder: str,
):
    """Display an interactive mask-plane viewer on a rotated mesh.

    Parameters
    ----------
    v, f : np.ndarray
        Mesh vertices and faces.
    T_rotation : np.ndarray
        3x3 rotation matrix (e.g. from interactive_rotation_widget).
    results_folder : str
        Directory for saving/loading transform_params.json.

    Returns
    -------
    state : dict
        Mutable dict with keys 'T_rotation', 'mask_plane_normal',
        'mask_plane_offset', 'mask_side', updated when user clicks Save.
    """
    centroid = v.mean(axis=0)
    v_c = v - centroid
    scale = np.abs(v_c).max()
    v_c = v_c / scale

    v_rotated = v_c @ T_rotation.T

    p2 = meshplot.plot(v_rotated, f, return_plot=True, shading={"width": 800, "height": 600})

    p2._renderer.controls[0].enableRotate = True
    p2._renderer.controls[0].enableZoom = True
    p2._renderer.controls[0].enablePan = True
    p2._renderer.camera.position = [0, 0, 3]
    p2._renderer.camera.up = [0, 1, 0]
    p2._renderer.controls[0].target = [0, 0, 0]

    axis_origin = np.array([-1.5, -1.5, -1.5])
    axis_length = 0.8
    for i, col in enumerate(["#ff0000", "#00ff00", "#0000ff"]):
        tip = axis_origin.copy()
        tip[i] += axis_length
        p2.add_lines(axis_origin, tip, shading={"line_color": col})

    def make_plane_mesh(normal, offset, size=1.5):
        n = np.array(normal, dtype=float)
        n = n / np.linalg.norm(n)
        if abs(n[2]) < 0.9:
            u = np.cross(n, [0, 0, 1])
        else:
            u = np.cross(n, [0, 1, 0])
        u = u / np.linalg.norm(u)
        w = np.cross(n, u)
        center = n * offset
        corners = np.array([
            center - size * u - size * w,
            center + size * u - size * w,
            center + size * u + size * w,
            center - size * u + size * w,
        ])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return corners, faces

    plane_v, plane_f = make_plane_mesh([1, 0, 0], 0)
    p2.add_mesh(plane_v, plane_f, c=np.array([1.0, 0.3, 0.3]))
    plane_obj_id = len(p2._Viewer__objects) - 1

    plane_nx = widgets.FloatSlider(min=-1, max=1, step=0.1, value=1, description='Normal X')
    plane_ny = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description='Normal Y')
    plane_nz = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description='Normal Z')
    plane_offset = widgets.FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Offset')
    side_toggle = widgets.ToggleButtons(options=['low', 'high'], value='low', description='Mask side')
    out2 = widgets.Output()

    btn_xy = widgets.Button(description="XY view")
    btn_xz = widgets.Button(description="XZ view")
    btn_yz = widgets.Button(description="YZ view")

    def set_view_xy(b):
        p2._renderer.camera.position = [0, 0, 3]
        p2._renderer.camera.up = [0, 1, 0]
        p2._renderer.controls[0].target = [0, 0, 0]

    def set_view_xz(b):
        p2._renderer.camera.position = [0, 3, 0]
        p2._renderer.camera.up = [0, 0, 1]
        p2._renderer.controls[0].target = [0, 0, 0]

    def set_view_yz(b):
        p2._renderer.camera.position = [3, 0, 0]
        p2._renderer.camera.up = [0, 1, 0]
        p2._renderer.controls[0].target = [0, 0, 0]

    btn_xy.on_click(set_view_xy)
    btn_xz.on_click(set_view_xz)
    btn_yz.on_click(set_view_yz)

    state = {
        "T_rotation": T_rotation.copy(),
        "mask_plane_normal": (1.0, 0.0, 0.0),
        "mask_plane_offset": 0.0,
        "mask_side": "low",
    }

    def update_plane(*args):
        normal = [plane_nx.value, plane_ny.value, plane_nz.value]
        pv, pf = make_plane_mesh(normal, plane_offset.value)
        p2.update_object(oid=plane_obj_id, vertices=pv)

        centroids = v_rotated[f].mean(axis=1)
        n = np.array(normal, dtype=float)
        n = n / np.linalg.norm(n)
        signed_dist = centroids @ n - plane_offset.value
        t = np.clip(signed_dist / 0.05, -5, 5)
        f_mask = 1.0 / (1.0 + np.exp(-t))
        f_mask[f_mask < 0.01] = 0.0
        f_mask[f_mask > 0.99] = 1.0
        if side_toggle.value == "low":
            f_mask = 1.0 - f_mask
        p2.update_object(oid=0, colors=f_mask)

        with out2:
            out2.clear_output()
            print(f"mask_plane_normal=({plane_nx.value}, {plane_ny.value}, {plane_nz.value})")
            print(f"mask_plane_offset={plane_offset.value}")
            print(f'mask_side="{side_toggle.value}"')

    for w in [plane_nx, plane_ny, plane_nz, plane_offset, side_toggle]:
        w.observe(update_plane, names='value')

    btn_save_plane = widgets.Button(description="Save plane params", button_style='info')

    def save_plane(b):
        state["mask_plane_normal"] = (plane_nx.value, plane_ny.value, plane_nz.value)
        state["mask_plane_offset"] = plane_offset.value
        state["mask_side"] = side_toggle.value
        with out2:
            out2.clear_output()
            print("Saved!")
            print(f"mask_plane_normal={state['mask_plane_normal']}")
            print(f"mask_plane_offset={state['mask_plane_offset']}")
            print(f'mask_side="{state["mask_side"]}"')

    btn_save_plane.on_click(save_plane)

    btn_save_disk = widgets.Button(description="Save to disk", button_style='warning')
    btn_load_disk = widgets.Button(description="Load from disk", button_style='')

    params_path = os.path.join(results_folder, 'transform_params.json')

    def save_to_disk(b):
        params = {
            "T_rotation": state["T_rotation"].tolist(),
            "mask_plane_normal": [plane_nx.value, plane_ny.value, plane_nz.value],
            "mask_plane_offset": plane_offset.value,
            "mask_side": side_toggle.value,
        }
        with open(params_path, 'w') as fp:
            json.dump(params, fp, indent=2)
        with out2:
            out2.clear_output()
            print(f"Saved to {params_path}")

    def load_from_disk(b):
        with open(params_path, 'r') as fp:
            params = json.load(fp)
        state["T_rotation"] = np.array(params["T_rotation"])
        state["mask_plane_normal"] = tuple(params["mask_plane_normal"])
        state["mask_plane_offset"] = params["mask_plane_offset"]
        state["mask_side"] = params["mask_side"]
        plane_nx.value = state["mask_plane_normal"][0]
        plane_ny.value = state["mask_plane_normal"][1]
        plane_nz.value = state["mask_plane_normal"][2]
        plane_offset.value = state["mask_plane_offset"]
        side_toggle.value = state["mask_side"]
        with out2:
            out2.clear_output()
            print(f"Loaded from {params_path}")
            print(f"T_rotation = np.array({state['T_rotation'].tolist()})")
            print(f"mask_plane_normal={state['mask_plane_normal']}")
            print(f"mask_plane_offset={state['mask_plane_offset']}")
            print(f'mask_side="{state["mask_side"]}"')

    btn_save_disk.on_click(save_to_disk)
    btn_load_disk.on_click(load_from_disk)

    display(widgets.VBox([
        plane_nx, plane_ny, plane_nz, plane_offset, side_toggle,
        widgets.HBox([btn_xy, btn_xz, btn_yz]),
        widgets.HBox([btn_save_plane, btn_save_disk, btn_load_disk]),
        out2
    ]))

    return state


def heal_single_mesh(
    v: np.ndarray,
    f: np.ndarray,
    output_path: str | None = None,
    clean_up_topology_iterations: int = 5,
    ensure_topo_mincomponentsize: int = 100,
    ensure_topo_maxhole: int = 2000,
    smooth_steps: int = 10,
    lambda_smooth: float = 0.8,
    mu_smooth: float = -0.83,
    target_remeshing_length: float = 10.0,
    remeshing_iterations: int = 5,
    min_face_area: float = 20.0,
    max_hole_size_cleanup: int = 500,
) -> dict:
    """Heal a single mesh: topology cleanup, smoothing, remeshing, export.

    Parameters
    ----------
    v, f : np.ndarray
        Input vertices and faces.
    output_path : str or None
        If provided, save the healed mesh as OBJ to this path.

    Returns
    -------
    dict with keys 'v' and 'f'.
    """
    # --- Step 1: Load into pymeshlab ---
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(v, f))

    # --- Step 2: Iterative topology cleanup ---
    for _ in range(clean_up_topology_iterations):
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_null_faces()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes(maxholesize=max_hole_size_cleanup)
        ms.meshing_remove_unreferenced_vertices()

    # --- Step 3: Coherent face winding ---
    ms.meshing_re_orient_faces_coherently()

    # --- Step 4: Euler characteristic check ---
    m = ms.current_mesh()
    V, F = m.vertex_number(), m.face_number()
    E = 3 * F // 2
    euler = V - E + F

    if euler != 2:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=ensure_topo_mincomponentsize,
        )
        ms.meshing_close_holes(maxholesize=ensure_topo_maxhole)

    # --- Step 5: Non-manifold repair ---
    m = ms.current_mesh()
    vfix, ffix = m.vertex_matrix(), m.face_matrix()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vfix, ffix))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    # --- Step 6: Taubin smoothing ---
    ms.apply_coord_taubin_smoothing(
        stepsmoothnum=smooth_steps,
        lambda_=lambda_smooth,
        mu=mu_smooth,
    )

    out = ms.current_mesh()
    v_clean = out.vertex_matrix()
    f_clean = out.face_matrix()

    # --- Step 7: Orient all normals outward ---
    components, _ = igl.orientable_patches(f_clean)
    f_clean, _ = igl.orient_outward(v_clean, f_clean, components)

    # --- Step 8: Isotropic remeshing ---
    ms.meshing_isotropic_explicit_remeshing(
        iterations=remeshing_iterations,
        targetlen=pymeshlab.PureValue(target_remeshing_length),
    )

    out = ms.current_mesh()
    v_clean = out.vertex_matrix()
    f_clean = out.face_matrix()

    # --- Step 9: Remove degenerate faces ---
    e1 = v_clean[f_clean[:, 1]] - v_clean[f_clean[:, 0]]
    e2 = v_clean[f_clean[:, 2]] - v_clean[f_clean[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    f_clean = f_clean[areas > min_face_area]

    used, remap = np.unique(f_clean, return_inverse=True)
    v_clean = v_clean[used]
    f_clean = remap.reshape(-1, 3)

    # Final hole closing
    v_clean, f_clean = pymeshfix.clean_from_arrays(v_clean, f_clean)

    # --- Step 10: Export OBJ ---
    if output_path is not None:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(v_clean, f_clean))
        ms.apply_normal_normalization_per_vertex()
        ms.save_current_mesh(output_path, save_vertex_normal=True)

    return {"v": v_clean, "f": f_clean}


def compute_face_mask(
    v: np.ndarray,
    f: np.ndarray,
    transform: np.ndarray,
    mask_plane_normal: tuple[float, float, float],
    mask_plane_offset: float,
    mask_side: str = "low",
    transition_width_fraction: float = 0.05,
    plot: bool = False,
) -> np.ndarray:
    """Compute a per-face sigmoid mask from a cutting plane.

    Parameters
    ----------
    v, f : np.ndarray
        Mesh vertices and faces.
    transform : np.ndarray
        3x3 rotation matrix applied to the mesh before masking.
    mask_plane_normal : tuple
        Normal vector of the cutting plane.
    mask_plane_offset : float
        Offset of the plane along the normal.
    mask_side : str
        'low' masks the negative side, 'high' masks the positive side.
    transition_width_fraction : float
        Controls the sharpness of the sigmoid transition.
    plot : bool
        If True, display the mesh coloured by the mask.

    Returns
    -------
    f_mask : np.ndarray of shape (n_faces,), values in [0, 1].
    """
    centroid = v.mean(axis=0)
    v_rot = (v - centroid) @ transform.T + centroid

    centroid_rot = v_rot.mean(axis=0)
    scale = np.abs(v_rot - centroid_rot).max()
    face_centroids = v_rot[f].mean(axis=1)
    face_centroids_norm = (face_centroids - centroid_rot) / scale

    normal = np.array(mask_plane_normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    signed_dist = face_centroids_norm @ normal - mask_plane_offset
    extent = np.abs(signed_dist).max()
    tw = transition_width_fraction * extent

    t = np.clip(signed_dist / tw, -5, 5)
    f_mask = 1.0 / (1.0 + np.exp(-t))
    f_mask[f_mask < 0.01] = 0.0
    f_mask[f_mask > 0.99] = 1.0
    if mask_side == "low":
        f_mask = 1.0 - f_mask

    if plot:
        meshplot.plot(v_rot, f, c=f_mask)

    return f_mask


def transform_single_mesh(
    v: np.ndarray,
    f: np.ndarray,
    transform: np.ndarray,
    mask_plane_normal: tuple[float, float, float],
    mask_plane_offset: float,
    mask_side: str = "low",
    transition_width_fraction: float = 0.05,
) -> dict:
    """Apply rotation and compute face mask for a single mesh.

    Returns
    -------
    dict with keys 'v' (rotated vertices), 'f', and 'f_mask'.
    """
    centroid = v.mean(axis=0)
    v_rot = (v - centroid) @ transform.T + centroid

    f_mask = compute_face_mask(
        v, f,
        transform=transform,
        mask_plane_normal=mask_plane_normal,
        mask_plane_offset=mask_plane_offset,
        mask_side=mask_side,
        transition_width_fraction=transition_width_fraction,
    )

    return {"v": v_rot, "f": f, "f_mask": f_mask}


def interactive_render_preview(
    v: np.ndarray,
    f: np.ndarray,
    scalars: np.ndarray | None = None,
    cmap: str = "coolwarm",
    clim: tuple[float, float] | None = None,
):
    """Interactive elev/azim picker using quick matplotlib renders.

    Adjust sliders and click 'Render' to preview. The render is saved
    to a temp file and displayed as an image widget (fast refresh).

    Returns a state dict with 'elev' and 'azim'.
    """
    import tempfile
    import io

    state = {"elev": 20.0, "azim": -60.0}

    elev_slider = widgets.IntSlider(min=-90, max=90, step=5, value=20, description='Elevation')
    azim_slider = widgets.IntSlider(min=-180, max=180, step=5, value=-60, description='Azimuth')
    btn_render = widgets.Button(description="Render", button_style='info')
    img_widget = widgets.Image(format='png', layout=widgets.Layout(width='800px'))
    label = widgets.Label(value="Click Render to preview")

    centroid = v.mean(axis=0)
    v_c = v - centroid

    tmp_path = os.path.join(tempfile.gettempdir(), "_preview_render.png")

    def do_render(b):
        state["elev"] = elev_slider.value
        state["azim"] = azim_slider.value
        render_single_frame(
            v_c, f, tmp_path,
            scalars=scalars, clim=clim, cmap=cmap,
            elev=state["elev"], azim=state["azim"], dpi=80,
        )
        with open(tmp_path, "rb") as fh:
            img_widget.value = fh.read()
        label.value = f"elev={state['elev']}, azim={state['azim']}"

    btn_render.on_click(do_render)
    do_render(None)

    display(widgets.VBox([
        widgets.HBox([elev_slider, azim_slider, btn_render]),
        label,
        img_widget,
    ]))

    return state


def render_single_frame(
    v: np.ndarray,
    f: np.ndarray,
    output_path: str,
    scalars: np.ndarray | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    cmap: str = "coolwarm",
    clim: tuple[float, float] | None = None,
):
    """Render a mesh to a PNG file using matplotlib (offscreen).

    Parameters
    ----------
    v, f : np.ndarray
        Mesh vertices and faces.
    output_path : str
        Path for the output PNG.
    scalars : np.ndarray or None
        Per-face scalar field for colouring.
    elev, azim : float
        Camera elevation and azimuth angles in degrees.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution.
    cmap : str
        Matplotlib colormap name.
    clim : tuple or None
        (vmin, vmax) for scalar colour range. Auto if None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import cm

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    verts_tri = v[f]  # (n_faces, 3, 3)

    if scalars is not None:
        colormap = cm.get_cmap(cmap)
        vmin = clim[0] if clim else scalars.min()
        vmax = clim[1] if clim else scalars.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        face_colors = colormap(norm(scalars))
    else:
        face_colors = "lightblue"

    poly = Poly3DCollection(verts_tri, linewidths=0.2, edgecolors="black")
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    # Auto-scale axes
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def make_movie(
    frames_folder: str,
    output_path: str,
    fps: int = 15,
    pattern: str = "*.png",
):
    """Stitch PNG frames into an mp4 video.

    Parameters
    ----------
    frames_folder : str
        Folder containing numbered PNG frames.
    output_path : str
        Output .mp4 path.
    fps : int
        Frames per second.
    pattern : str
        Glob pattern for frame files (sorted alphabetically).
    """
    import glob as _glob
    import imageio

    files = sorted(_glob.glob(os.path.join(frames_folder, pattern)))
    writer = imageio.get_writer(output_path, fps=fps)
    for fpath in files:
        frame = imageio.imread(fpath)
        writer.append_data(frame)
    writer.close()

# aling with point clouds
def rotation_to_o3d_transform(rot: Rotation) -> np.ndarray:
    """Convert scipy Rotation → 4×4 homogeneous matrix for open3d."""
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    return T


def o3d_transform_to_rotation(T: np.ndarray) -> Rotation:
    """Extract and re-orthogonalise rotation from a 4×4 open3d transform."""
    R_mat = T[:3, :3]
    U, _, Vt = np.linalg.svd(R_mat)   # re-orthogonalise after floating-point drift
    return Rotation.from_matrix(U @ Vt)


def make_point_cloud(verts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.estimate_normals()
    return pcd

def normalize(verts: np.ndarray) -> np.ndarray:
    verts = verts - verts.mean(axis=0)          # centre
    verts = verts / np.linalg.norm(verts, axis=1).max()  # scale to unit sphere
    return verts

def run_icp(
    source_verts: np.ndarray,
    target_verts: np.ndarray,
    init_rot: Rotation = None,
    max_correspondence_distance: float = 0.1,   # now in normalised units, ~0.05–0.2
    max_iteration: int = 200,
) -> Rotation:
    source = make_point_cloud(normalize(source_verts))   # ← normalise
    target = make_point_cloud(normalize(target_verts))   # ← normalise

    init_T = rotation_to_o3d_transform(init_rot) if init_rot is not None else np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_correspondence_distance,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )
    return o3d_transform_to_rotation(result.transformation)

def interactive_alignment_widget(v_ref, f_ref, c_ref,
                                 v_target, f_target, c_target,
                                 init_rotation=None,
                                 panel_size=(450, 500)):
    """Side-by-side reference (left) + rotated target (right), both colored by `c`.
    Sliders rotate the target; click 'Save T_rotation' to record the matrix.

    Convention: target is displayed as v_target @ T, matching `v_trg @ rot` elsewhere.
    Save with: spatial_rotations[name] = Rotation.from_matrix(state['T_rotation'])
    """
    # Center each independently; common scale
    v_ref_c    = v_ref    - v_ref.mean(axis=0)
    v_target_c = v_target - v_target.mean(axis=0)
    scale = max(np.abs(v_ref_c).max(), np.abs(v_target_c).max())
    v_ref_c    /= scale
    v_target_c /= scale

    T_init = np.eye(3) if init_rotation is None else np.asarray(init_rotation)

    out_ref    = widgets.Output()
    out_target = widgets.Output()
    out_log    = widgets.Output()

    width, height = panel_size
    shading = {"width": width, "height": height}

    with out_ref:
        p_ref = meshplot.plot(v_ref_c, f_ref, c=c_ref, return_plot=True, shading=shading)
    with out_target:
        p_target = meshplot.plot(v_target_c @ T_init, f_target, c=c_target,
                                 return_plot=True, shading=shading)

    # Initial camera on both
    def _set_view(plot, pos, up):
        plot._renderer.camera.position = pos
        plot._renderer.camera.up       = up
        plot._renderer.controls[0].target = [0, 0, 0]
    for plot in (p_ref, p_target):
        plot._renderer.controls[0].enableRotate = True
        plot._renderer.controls[0].enableZoom   = True
        plot._renderer.controls[0].enablePan    = True
        _set_view(plot, [0, 0, 3], [0, 1, 0])

    rot_x = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot X', continuous_update=True)
    rot_y = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot Y', continuous_update=True)
    rot_z = widgets.FloatSlider(min=-180, max=180, step=1, value=0, description='Rot Z', continuous_update=True)
    flip_x = widgets.Checkbox(value=False, description='Flip X')
    flip_y = widgets.Checkbox(value=False, description='Flip Y')
    flip_z = widgets.Checkbox(value=False, description='Flip Z')

    state = {"T_baked": T_init.copy(), "T_rotation": T_init.copy()}

    def get_current_T():
        R = Rotation.from_euler('xyz', [rot_x.value, rot_y.value, rot_z.value], degrees=True).as_matrix()
        S = np.diag([-1.0 if flip_x.value else 1.0,
                     -1.0 if flip_y.value else 1.0,
                     -1.0 if flip_z.value else 1.0])
        return S @ R

    def update(*args):
        T_total = get_current_T() @ state["T_baked"]
        p_target.update_object(vertices=v_target_c @ T_total)
        with out_log:
            out_log.clear_output()
            print(f"T_rotation = np.array({np.round(T_total, 6).tolist()})")

    def bake(b):
        state["T_baked"] = get_current_T() @ state["T_baked"]
        for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
            w.unobserve(update, names='value')
        rot_x.value = rot_y.value = rot_z.value = 0
        flip_x.value = flip_y.value = flip_z.value = False
        for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
            w.observe(update, names='value')
        with out_log:
            out_log.clear_output()
            print("Baked!")
            print(f"T_baked = np.array({np.round(state['T_baked'], 6).tolist()})")

    def reset_all(b):
        state["T_baked"] = np.eye(3)
        bake(b)

    def save_T(b):
        state["T_rotation"] = get_current_T() @ state["T_baked"]
        with out_log:
            out_log.clear_output()
            print("Saved!")
            print(f"T_rotation = np.array({np.round(state['T_rotation'], 6).tolist()})")

    btn_bake  = widgets.Button(description="Bake rotation",   button_style='success')
    btn_reset = widgets.Button(description="Reset all",       button_style='danger')
    btn_save  = widgets.Button(description="Save T_rotation", button_style='info')
    btn_bake.on_click(bake); btn_reset.on_click(reset_all); btn_save.on_click(save_T)

    # View buttons sync BOTH panels
    def view_xy(b):
        for plot in (p_ref, p_target): _set_view(plot, [0, 0, 3], [0, 1, 0])
    def view_xz(b):
        for plot in (p_ref, p_target): _set_view(plot, [0, 3, 0], [0, 0, 1])
    def view_yz(b):
        for plot in (p_ref, p_target): _set_view(plot, [3, 0, 0], [0, 1, 0])

    btn_xy = widgets.Button(description="XY view")
    btn_xz = widgets.Button(description="XZ view")
    btn_yz = widgets.Button(description="YZ view")
    btn_xy.on_click(view_xy); btn_xz.on_click(view_xz); btn_yz.on_click(view_yz)

    for w in [rot_x, rot_y, rot_z, flip_x, flip_y, flip_z]:
        w.observe(update, names='value')

    label_ref    = widgets.HTML("<b>Reference</b>")
    label_target = widgets.HTML("<b>Target (rotated)</b>")

    display(widgets.VBox([
        widgets.HBox([widgets.VBox([label_ref, out_ref]),
                      widgets.VBox([label_target, out_target])]),
        widgets.HBox([rot_x, rot_y, rot_z]),
        widgets.HBox([flip_x, flip_y, flip_z]),
        widgets.HBox([btn_bake, btn_reset, btn_save]),
        widgets.HBox([btn_xy, btn_xz, btn_yz]),
        out_log,
    ]))

    return state