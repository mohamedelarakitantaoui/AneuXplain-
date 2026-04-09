"""
morphology_analyzer.py - Clinical Morphological Parameter Extraction

Extracts 8 geometric measurements used by neurosurgeons to evaluate
intracranial aneurysm rupture risk from 3D mesh surfaces.

Parameters: Neck Width, Dome Height, Aspect Ratio, Max Dome Diameter,
Dome-to-Neck Ratio, Irregularity Index, Volume, Surface Area.
"""

from collections import deque

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist


# ============================================
# Visualization colors for each measurement
# ============================================
COLORS = {
    "neck_width": "#00BFFF",       # deep sky blue
    "dome_height": "#FF6B35",      # orange
    "aspect_ratio": "#FFD700",     # gold
    "max_dome_diameter": "#FF1493", # deep pink
    "dome_to_neck": "#8A2BE2",     # blue violet
    "irregularity": "#FF4444",     # red
    "volume": "#32CD32",           # lime green
    "surface_area": "#1E90FF",     # dodger blue
}


def _measurement(name, value, unit, status, spatial, reason=None):
    """Build a standardized measurement dict."""
    entry = {
        "name": name,
        "value": value,
        "unit": unit,
        "status": status,
        "spatial": spatial,
    }
    if reason is not None:
        entry["reason"] = reason
    return entry


def _failed(name, unit, reason):
    """Build a failed measurement dict."""
    return _measurement(
        name, None, unit, "failed",
        spatial={"vertex_indices": [], "vertex_positions": [], "type": "point", "color": "#888888"},
        reason=reason,
    )


# ============================================
# Ellipsoid fitting helper
# ============================================
def _fit_ellipsoid(points):
    """
    Fit an ellipsoid to a point cloud using the algebraic method.

    Centers and scales the data first for numerical stability, then
    solves the 9-parameter quadratic surface:
        Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

    Returns (center, radii, rotation) or None on failure.
    Falls back to a PCA-based ellipsoid if the algebraic fit yields
    an invalid quadric (e.g. hyperboloid).
    """
    # Center and scale for numerical stability
    centroid = points.mean(axis=0)
    pts = points - centroid
    scale = np.std(pts)
    if scale < 1e-12:
        return None
    pts = pts / scale

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Build the design matrix
    D = np.column_stack([
        x * x, y * y, z * z,
        x * y, x * z, y * z,
        x, y, z,
    ])

    # Solve D @ v = 1  (least-squares)
    ones = np.ones(len(pts))
    result, residuals, rank, sv = np.linalg.lstsq(D, ones, rcond=None)
    if rank < 9:
        return _fit_ellipsoid_pca(points)

    A, B, C, D_coeff, E, F, G, H, I = result

    Q33 = np.array([
        [A,        D_coeff/2, E/2],
        [D_coeff/2, B,        F/2],
        [E/2,       F/2,      C  ],
    ])
    q = np.array([G/2, H/2, I/2])

    try:
        center_local = -np.linalg.solve(Q33, q)
    except np.linalg.LinAlgError:
        return _fit_ellipsoid_pca(points)

    offset = -1.0 + q @ np.linalg.solve(Q33, q)
    if abs(offset) < 1e-12:
        return _fit_ellipsoid_pca(points)

    M = -Q33 / offset
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    if np.any(eigenvalues <= 0):
        return _fit_ellipsoid_pca(points)

    radii_local = 1.0 / np.sqrt(eigenvalues)

    # Un-scale back to original coordinate space
    center = center_local * scale + centroid
    radii = radii_local * scale
    rotation = eigenvectors

    return center, radii, rotation


def _fit_ellipsoid_pca(points):
    """
    Fallback: fit an axis-aligned ellipsoid in PCA space.

    Uses the covariance structure of the point cloud. The semi-axes
    are set to 2 * sqrt(eigenvalue) so the ellipsoid roughly encloses
    the cloud (covers ~95% under a Gaussian assumption).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(eigenvalues <= 0):
        return None

    radii = 2.0 * np.sqrt(eigenvalues)
    return centroid, radii, eigenvectors


def _ellipsoid_distance(points, center, radii, rotation):
    """
    Compute unsigned distance from each point to the ellipsoid surface.

    Projects each point into the ellipsoid's local coordinate frame,
    scales to a unit sphere, computes the radial distance, then scales
    back to the original space.
    """
    local = (points - center) @ rotation  # into ellipsoid frame
    # Normalized coordinates (on unit sphere, point would be at radius 1)
    normalized = local / radii
    r = np.linalg.norm(normalized, axis=1)
    # Distance from surface ≈ |r - 1| * mean_radius (approximate but fast)
    mean_radius = np.mean(radii)
    return np.abs(r - 1.0) * mean_radius


# ============================================
# MorphologyAnalyzer
# ============================================
class MorphologyAnalyzer:
    """
    Extracts clinical morphological parameters from an aneurysm mesh.

    Usage:
        analyzer = MorphologyAnalyzer()
        result = analyzer.analyze("path/to/aneurysm.obj")
    """

    def __init__(self, neck_slices=100, dome_slices=50):
        """
        Args:
            neck_slices: Number of cross-sectional slices for closed-mesh
                         neck detection.
            dome_slices: Number of parallel slices for max dome diameter.
        """
        self.neck_slices = neck_slices
        self.dome_slices = dome_slices

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def analyze(self, mesh_or_path) -> dict:
        """
        Analyze a mesh and return 8 clinical morphological parameters.

        Args:
            mesh_or_path: A trimesh.Trimesh object or a file path string.

        Returns:
            dict with keys: measurements, neck_plane, dome_vertices,
            neck_vertices.
        """
        if isinstance(mesh_or_path, str):
            mesh = trimesh.load(mesh_or_path, force="mesh")
        else:
            mesh = mesh_or_path

        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces)

        # ---- Step 1: Detect the neck ----
        neck_info = self._detect_neck(mesh, vertices, faces)
        if neck_info is None:
            return self._all_failed(
                "Could not detect neck plane — mesh may be degenerate"
            )

        neck_point, neck_normal, neck_boundary_indices, dome_mask = neck_info

        # neck_normal already points toward the dome (set by _detect_neck)
        signed_dists = (vertices - neck_point) @ neck_normal
        dome_indices = np.where(dome_mask)[0].tolist()

        # ---- Step 2: Compute each measurement ----
        measurements = []

        # b) Neck Width
        nw_result = self._neck_width(vertices, neck_boundary_indices)
        measurements.append(nw_result)
        neck_width = nw_result["value"]

        # c) Dome Height
        dh_result = self._dome_height(
            vertices, signed_dists, dome_mask, neck_boundary_indices
        )
        measurements.append(dh_result)
        dome_height = dh_result["value"]

        # d) Aspect Ratio
        measurements.append(
            self._aspect_ratio(dome_height, neck_width, dh_result.get("spatial"))
        )

        # e) Max Dome Diameter
        mdd_result = self._max_dome_diameter(
            mesh, vertices, neck_point, neck_normal, dome_height, dome_mask
        )
        measurements.append(mdd_result)
        max_dome_diam = mdd_result["value"]

        # f) Dome-to-Neck Ratio
        measurements.append(
            self._dome_to_neck_ratio(max_dome_diam, neck_width,
                                     mdd_result.get("spatial"),
                                     nw_result.get("spatial"))
        )

        # g) Irregularity Index
        measurements.append(
            self._irregularity_index(vertices, dome_mask, dome_indices)
        )

        # h) Volume
        measurements.append(
            self._volume(mesh, vertices, faces, neck_point, neck_normal,
                         signed_dists, neck_boundary_indices)
        )

        # i) Surface Area
        measurements.append(
            self._surface_area(vertices, faces, dome_mask)
        )

        return {
            "measurements": measurements,
            "neck_plane": {
                "point": neck_point.tolist(),
                "normal": neck_normal.tolist(),
            },
            "dome_vertices": dome_indices,
            "neck_vertices": [int(i) for i in neck_boundary_indices],
        }

    # --------------------------------------------------
    # Neck detection  (three-tier cascade)
    # --------------------------------------------------
    def _detect_neck(self, mesh, vertices, faces):
        """
        Detect the aneurysm neck plane.

        Three strategies, tried in order:
          1. Centerline-based: identify the aneurysm dome as the lateral
             bulge off the parent vessel, then find the neck ring where
             the dome meets the vessel surface.
          2. Curvature-based: find the band of high-negative Gaussian
             curvature (saddle geometry) away from open boundaries.
          3. Boundary-based (fallback): for isolated dome meshes, use
             open boundary edges as the neck.

        Returns (neck_center, neck_normal, neck_indices, dome_mask)
        where neck_normal already points toward the dome side and
        dome_mask is a boolean array marking the dome vertices.
        Returns None on failure.
        """
        adjacency = self._build_adjacency(faces)

        # Strategy 1: centerline-based radial bulge detection
        result = self._neck_by_centerline(vertices, faces, adjacency)
        if result is not None:
            return result

        # Strategy 2: curvature-based saddle ring detection
        result = self._neck_by_curvature(mesh, vertices, faces, adjacency)
        if result is not None:
            return result

        # Strategy 3: open boundary fallback (isolated dome meshes)
        boundary_indices = self._find_boundary_vertices(faces)
        if len(boundary_indices) >= 3:
            return self._neck_from_boundary(vertices, boundary_indices)

        return None

    # -- helpers shared by all strategies --

    @staticmethod
    def _build_adjacency(faces):
        adj = {}
        for f in faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        return adj

    @staticmethod
    def _find_boundary_vertices(faces):
        edge_count = {}
        for f in faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1
        bv = set()
        for (v0, v1), c in edge_count.items():
            if c == 1:
                bv.add(v0)
                bv.add(v1)
        return np.array(sorted(bv), dtype=int)

    @staticmethod
    def _connected_components(mask, adjacency):
        """Vertex-connected components within *mask*."""
        visited = set()
        components = []
        for v in np.where(mask)[0]:
            if v in visited:
                continue
            comp = []
            queue = deque([v])
            while queue:
                u = queue.popleft()
                if u in visited or not mask[u]:
                    continue
                visited.add(u)
                comp.append(u)
                for nb in adjacency.get(u, []):
                    if nb not in visited and mask[nb]:
                        queue.append(nb)
            if comp:
                components.append(np.array(comp))
        return sorted(components, key=len, reverse=True)

    def _neck_ring_from_dome(self, vertices, dome_mask, adjacency):
        """
        Given a dome vertex mask, find the ring of non-dome vertices
        that are mesh-adjacent to at least one dome vertex.
        Fit a plane and orient the normal toward the dome.

        Also grows the dome by one ring (includes the neck ring vertices
        on the dome side) so the dome mask tightly wraps the actual
        bulge rather than everything above the plane.
        """
        neck_list = []
        for vi in range(len(vertices)):
            if dome_mask[vi]:
                continue
            for nb in adjacency.get(vi, []):
                if dome_mask[nb]:
                    neck_list.append(vi)
                    break
        if len(neck_list) < 3:
            return None
        neck_indices = np.array(neck_list, dtype=int)

        npts = vertices[neck_indices]
        center = npts.mean(axis=0)
        _, _, Vt = np.linalg.svd(npts - center, full_matrices=False)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)

        # Orient normal toward the dome
        dome_centroid = vertices[dome_mask].mean(axis=0)
        if (dome_centroid - center) @ normal < 0:
            normal = -normal

        # Build a tight dome mask: the dome component + neck ring
        tight_dome = dome_mask.copy()
        signed = (vertices - center) @ normal
        for vi in neck_list:
            if signed[vi] > 0:
                tight_dome[vi] = True

        return center, normal, neck_indices, tight_dome

    # -- Strategy 1: centerline-based --

    def _neck_by_centerline(self, vertices, faces, adjacency):
        """
        Find the dome as the lateral bulge off a tubular parent vessel.

        1. Compute the vessel's principal axis.
        2. Compute each vertex's radial distance from that axis.
        3. Threshold (median + k*MAD) to isolate the bulge, sweeping
           k from aggressive (3.0) to loose (1.5).
        4. Keep the largest connected component as the dome.
        5. Neck = non-dome vertices adjacent to dome vertices.
        """
        centered = vertices - vertices.mean(axis=0)
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        vessel_axis = Vt[0]

        # Elongation check: if the mesh isn't clearly tubular
        # (PC1 ≫ PC2), the centerline approach may not apply.
        if S[0] < 1.5 * S[1]:
            return None

        proj = np.outer(centered @ vessel_axis, vessel_axis)
        radial = np.linalg.norm(centered - proj, axis=1)

        median_r = np.median(radial)
        mad = np.median(np.abs(radial - median_r))
        if mad < 1e-12:
            return None

        for k in [3.0, 2.5, 2.0, 1.5]:
            threshold = median_r + k * mad
            dome_mask = radial > threshold
            if dome_mask.sum() < 5:
                continue

            comps = self._connected_components(dome_mask, adjacency)
            if not comps:
                continue

            largest = comps[0]
            frac = len(largest) / len(vertices)
            if not (0.005 < frac < 0.40):
                continue

            dome = np.zeros(len(vertices), dtype=bool)
            dome[largest] = True

            result = self._neck_ring_from_dome(vertices, dome, adjacency)
            if result is not None:
                return result

        return None

    # -- Strategy 2: curvature-based --

    def _neck_by_curvature(self, mesh, vertices, faces, adjacency):
        """
        Find the neck as the band of high-negative Gaussian curvature
        (saddle-point geometry) away from the mesh's open boundaries.
        """
        boundary = self._find_boundary_vertices(faces)

        avg_edge = np.mean([
            np.linalg.norm(vertices[e[0]] - vertices[e[1]])
            for e in mesh.edges_unique[:200]
        ])

        # Gaussian curvature
        gc = trimesh.curvature.discrete_gaussian_curvature_measure(
            mesh, mesh.vertices, radius=avg_edge * 3
        )

        # Mask out boundary-adjacent vertices
        if len(boundary) > 0:
            btree = cKDTree(vertices[boundary])
            dist_to_boundary, _ = btree.query(vertices)
            interior = dist_to_boundary > avg_edge * 5
        else:
            interior = np.ones(len(vertices), dtype=bool)

        if interior.sum() < 20:
            return None

        gc_interior = gc[interior]
        thr = np.percentile(gc_interior, 5)
        if thr >= 0:
            return None  # no meaningful negative curvature

        saddle_mask = (gc < thr) & interior
        if saddle_mask.sum() < 3:
            return None

        # The saddle vertices should lie between the dome and the vessel.
        # Use them to define a plane, then the dome is the smaller side.
        saddle_pts = vertices[saddle_mask]
        center = saddle_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(saddle_pts - center, full_matrices=False)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)

        # Orient normal toward the smaller side (= dome)
        signed = (vertices - center) @ normal
        if np.sum(signed > 0) > np.sum(signed < 0):
            normal = -normal
            signed = -signed

        dome_mask = signed > 0
        neck_indices = np.where(saddle_mask)[0]
        return center, normal, neck_indices, dome_mask

    # -- Strategy 3: boundary fallback --

    @staticmethod
    def _neck_from_boundary(vertices, boundary_indices):
        """For isolated dome meshes whose open boundary IS the neck."""
        bpts = vertices[boundary_indices]
        center = bpts.mean(axis=0)
        _, _, Vt = np.linalg.svd(bpts - center, full_matrices=False)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)

        # Orient toward the majority of the mesh (= dome for isolated domes)
        signed = (vertices - center) @ normal
        if np.sum(signed > 0) < np.sum(signed < 0):
            normal = -normal
            signed = -signed

        dome_mask = signed > 0
        return center, normal, boundary_indices, dome_mask

    # --------------------------------------------------
    # Individual measurements
    # --------------------------------------------------
    def _neck_width(self, vertices, neck_indices):
        """b) Maximum distance between any two neck boundary vertices."""
        if len(neck_indices) < 2:
            return _failed("Neck Width", "mm", "Fewer than 2 neck vertices")

        pts = vertices[neck_indices]
        dists = cdist(pts, pts)
        idx = np.unravel_index(np.argmax(dists), dists.shape)
        width = dists[idx]
        v_a = int(neck_indices[idx[0]])
        v_b = int(neck_indices[idx[1]])

        return _measurement(
            "Neck Width", float(width), "mm", "computed",
            spatial={
                "vertex_indices": [v_a, v_b],
                "vertex_positions": [vertices[v_a].tolist(), vertices[v_b].tolist()],
                "type": "line",
                "color": COLORS["neck_width"],
            },
        )

    def _dome_height(self, vertices, signed_dists, dome_mask, neck_indices):
        """c) Maximum perpendicular distance from the neck plane to the dome apex."""
        if not np.any(dome_mask):
            return _failed("Dome Height", "mm", "No vertices above neck plane")

        # Only consider dome vertices for the apex
        dome_dists = np.where(dome_mask, signed_dists, -np.inf)
        apex_local = np.argmax(dome_dists)
        height = float(signed_dists[apex_local])

        neck_center = vertices[neck_indices].mean(axis=0)

        return _measurement(
            "Dome Height", height, "mm", "computed",
            spatial={
                "vertex_indices": [int(apex_local)],
                "plane_vertices": [int(i) for i in neck_indices],
                "vertex_positions": [vertices[apex_local].tolist(), neck_center.tolist()],
                "type": "line",
                "color": COLORS["dome_height"],
            },
        )

    def _aspect_ratio(self, dome_height, neck_width, dh_spatial=None):
        """d) aspect_ratio = dome_height / neck_width."""
        if dome_height is None or neck_width is None or neck_width == 0:
            return _failed("Aspect Ratio", "", "Missing dome height or neck width")

        ar = dome_height / neck_width
        # Reuse dome height spatial (aspect ratio is derived from height/width)
        positions = dh_spatial.get("vertex_positions", []) if dh_spatial else []
        return _measurement(
            "Aspect Ratio", float(ar), "", "computed",
            spatial={
                "vertex_indices": dh_spatial.get("vertex_indices", []) if dh_spatial else [],
                "vertex_positions": positions,
                "type": "line" if len(positions) >= 2 else "point",
                "color": COLORS["aspect_ratio"],
            },
        )

    def _max_dome_diameter(self, mesh, vertices, neck_point, neck_normal,
                           dome_height, dome_mask):
        """
        e) Maximum width of the dome, measured parallel to the neck plane.

        Only considers vertices within the dome region (not the parent
        vessel on the same side of the plane).
        """
        if dome_height is None or dome_height <= 0:
            return _failed("Max Dome Diameter", "mm", "Invalid dome height")

        best_diam = 0.0
        best_pair = (0, 0)

        heights = np.linspace(0.05 * dome_height, 0.95 * dome_height,
                              self.dome_slices)

        signed_dists = (vertices - neck_point) @ neck_normal

        for h in heights:
            tolerance = dome_height / self.dome_slices
            band_mask = (np.abs(signed_dists - h) < tolerance) & dome_mask
            band_indices = np.where(band_mask)[0]
            if len(band_indices) < 2:
                continue

            pts = vertices[band_indices]
            dists = cdist(pts, pts)
            idx = np.unravel_index(np.argmax(dists), dists.shape)
            d = dists[idx]
            if d > best_diam:
                best_diam = d
                best_pair = (
                    int(band_indices[idx[0]]),
                    int(band_indices[idx[1]]),
                )

        if best_diam == 0:
            return _failed("Max Dome Diameter", "mm",
                           "Could not find valid dome cross-sections")

        return _measurement(
            "Max Dome Diameter", float(best_diam), "mm", "computed",
            spatial={
                "vertex_indices": [best_pair[0], best_pair[1]],
                "vertex_positions": [vertices[best_pair[0]].tolist(), vertices[best_pair[1]].tolist()],
                "type": "line",
                "color": COLORS["max_dome_diameter"],
            },
        )

    def _dome_to_neck_ratio(self, max_dome_diam, neck_width,
                             mdd_spatial=None, nw_spatial=None):
        """f) dome_to_neck = max_dome_diameter / neck_width."""
        if max_dome_diam is None or neck_width is None or neck_width == 0:
            return _failed("Dome-to-Neck Ratio", "",
                           "Missing dome diameter or neck width")

        ratio = max_dome_diam / neck_width
        # Combine positions from max dome diameter and neck width
        positions = []
        if mdd_spatial and "vertex_positions" in mdd_spatial:
            positions.extend(mdd_spatial["vertex_positions"])
        if nw_spatial and "vertex_positions" in nw_spatial:
            positions.extend(nw_spatial["vertex_positions"])
        return _measurement(
            "Dome-to-Neck Ratio", float(ratio), "", "computed",
            spatial={
                "vertex_indices": [],
                "vertex_positions": positions,
                "type": "region" if positions else "point",
                "color": COLORS["dome_to_neck"],
            },
        )

    def _irregularity_index(self, vertices, dome_mask, dome_indices):
        """
        g) Measures surface deviation from a smooth ellipsoid.

        irregularity_index = std(distances) / mean(distances)
        """
        if np.sum(dome_mask) < 20:
            return _failed("Irregularity Index", "",
                           "Too few dome vertices for ellipsoid fitting")

        dome_pts = vertices[dome_mask]

        fit = _fit_ellipsoid(dome_pts)
        if fit is None:
            return _failed("Irregularity Index", "",
                           "Ellipsoid fitting failed — degenerate geometry")

        center, radii, rotation = fit
        dists = _ellipsoid_distance(dome_pts, center, radii, rotation)

        mean_dist = np.mean(dists)
        if mean_dist < 1e-12:
            return _failed("Irregularity Index", "",
                           "All dome vertices lie on the ellipsoid")

        ii = float(np.std(dists) / mean_dist)

        # Top 10% most irregular vertices
        n_top = max(1, int(0.1 * len(dists)))
        top_local = np.argsort(dists)[-n_top:]
        dome_idx_array = np.array(dome_indices)
        top_global = dome_idx_array[top_local].tolist()
        top_positions = dome_pts[top_local].tolist()

        return _measurement(
            "Irregularity Index", ii, "", "computed",
            spatial={
                "vertex_indices": top_global,
                "vertex_positions": top_positions,
                "type": "region",
                "color": COLORS["irregularity"],
            },
        )

    def _volume(self, mesh, vertices, faces, neck_point, neck_normal,
                signed_dists, neck_boundary_indices):
        """
        h) Volume of the aneurysm dome above the neck plane.

        Caps the dome at the neck boundary, then uses the divergence
        theorem (signed tetrahedra). Falls back to convex hull if the
        result is not watertight.
        """
        dome_mask = signed_dists > 0

        # Collect faces that have at least one vertex above the neck plane
        dome_face_mask = dome_mask[faces].any(axis=1)
        dome_faces = faces[dome_face_mask]

        if len(dome_faces) < 4:
            return _failed("Volume", "mm³", "Too few dome faces")

        # Project dome vertices that are below the plane up to the plane
        # (clip them) so we have a clean cut
        clipped_verts = vertices.copy()
        below_in_dome_faces = set(dome_faces.ravel()) - set(
            np.where(dome_mask)[0]
        )
        for vi in below_in_dome_faces:
            # Project onto neck plane
            d = signed_dists[vi]
            clipped_verts[vi] = vertices[vi] - d * neck_normal

        # Build a cap over the neck opening
        cap_faces = self._triangulate_boundary(
            clipped_verts, neck_boundary_indices, neck_point, neck_normal
        )

        if cap_faces is not None and len(cap_faces) > 0:
            all_faces = np.vstack([dome_faces, cap_faces])
        else:
            all_faces = dome_faces

        # Divergence theorem: V = (1/6) |Σ (v0 · (v1 × v2))|
        vol = 0.0
        for f in all_faces:
            v0 = clipped_verts[f[0]]
            v1 = clipped_verts[f[1]]
            v2 = clipped_verts[f[2]]
            vol += np.dot(v0, np.cross(v1, v2))
        vol = abs(vol) / 6.0

        # Sanity check — if volume is near zero, try convex hull fallback
        if vol < 1e-12:
            try:
                dome_pts = vertices[dome_mask]
                hull = ConvexHull(dome_pts)
                vol = hull.volume
            except Exception:
                return _failed("Volume", "mm³",
                               "Volume computation failed")

        dome_indices = np.where(dome_mask)[0].tolist()
        dome_positions = vertices[dome_mask][:100].tolist()
        return _measurement(
            "Volume", float(vol), "mm³", "computed",
            spatial={
                "vertex_indices": dome_indices[:100],  # cap for payload size
                "vertex_positions": dome_positions,
                "type": "region",
                "color": COLORS["volume"],
            },
        )

    def _surface_area(self, vertices, faces, dome_mask):
        """
        i) Surface area of the dome above the neck plane.

        Sums triangle areas for faces with at least one vertex above the
        neck plane.
        """
        dome_face_mask = dome_mask[faces].any(axis=1)
        dome_faces = faces[dome_face_mask]

        if len(dome_faces) == 0:
            return _failed("Surface Area", "mm²", "No dome faces found")

        v0 = vertices[dome_faces[:, 0]]
        v1 = vertices[dome_faces[:, 1]]
        v2 = vertices[dome_faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        sa = float(np.sum(areas))

        dome_indices = np.where(dome_mask)[0].tolist()
        dome_positions = vertices[dome_mask][:100].tolist()
        return _measurement(
            "Surface Area", float(sa), "mm²", "computed",
            spatial={
                "vertex_indices": dome_indices[:100],
                "vertex_positions": dome_positions,
                "type": "region",
                "color": COLORS["surface_area"],
            },
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _triangulate_boundary(self, vertices, boundary_indices, center,
                              normal):
        """
        Fan-triangulate a boundary loop from its centroid.

        Returns an (M, 3) array of face indices, or None on failure.
        """
        if len(boundary_indices) < 3:
            return None

        pts = vertices[boundary_indices]
        centroid = pts.mean(axis=0)

        # Order boundary vertices by angle around the centroid
        local = pts - centroid
        # Build a 2D basis on the neck plane
        u = local[0] - np.dot(local[0], normal) * normal
        u_len = np.linalg.norm(u)
        if u_len < 1e-12:
            u = local[1] - np.dot(local[1], normal) * normal
            u_len = np.linalg.norm(u)
        if u_len < 1e-12:
            return None
        u = u / u_len
        v = np.cross(normal, u)

        angles = np.arctan2(local @ v, local @ u)
        order = np.argsort(angles)
        ordered = boundary_indices[order]

        # Add the centroid as a new virtual vertex
        # Since we can't add vertices, use fan triangulation between
        # successive boundary vertices and the first boundary vertex
        # Actually, for a proper cap we do ear-clipping style fan:
        n = len(ordered)
        cap_faces = []
        anchor = ordered[0]
        for i in range(1, n - 1):
            # Wind so that the cap normal faces inward (opposite to dome)
            cap_faces.append([anchor, ordered[i + 1], ordered[i]])

        return np.array(cap_faces, dtype=int) if cap_faces else None

    def _all_failed(self, reason):
        """Return a result dict where every measurement is failed."""
        names_units = [
            ("Neck Width", "mm"),
            ("Dome Height", "mm"),
            ("Aspect Ratio", ""),
            ("Max Dome Diameter", "mm"),
            ("Dome-to-Neck Ratio", ""),
            ("Irregularity Index", ""),
            ("Volume", "mm³"),
            ("Surface Area", "mm²"),
        ]
        return {
            "measurements": [_failed(n, u, reason) for n, u in names_units],
            "neck_plane": None,
            "dome_vertices": [],
            "neck_vertices": [],
        }


# ============================================
# Standalone test
# ============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python morphology_analyzer.py <mesh.obj>")
        sys.exit(1)

    analyzer = MorphologyAnalyzer()
    result = analyzer.analyze(sys.argv[1])

    if result["neck_plane"] is not None:
        p = result["neck_plane"]["point"]
        n = result["neck_plane"]["normal"]
        print(f"Neck plane  — center: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}), "
              f"normal: ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})")
        print(f"Dome vertices: {len(result['dome_vertices'])}, "
              f"Neck vertices: {len(result['neck_vertices'])}")
    print()

    for m in result["measurements"]:
        if m["status"] == "computed":
            print(f"  {m['name']:>22s}: {m['value']:>10.3f} {m['unit']:<4s}  "
                  f"[{m['status']}]")
        else:
            reason = m.get("reason", "unknown")
            print(f"  {m['name']:>22s}: {'—':>10s}       "
                  f"[{m['status']}: {reason}]")
