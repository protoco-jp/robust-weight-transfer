# This file is part of Robust Weight Transfer for Blender.
#
# Portions of this code are based on:
#   RobustSkinWeightsTransferCode (https://github.com/rin-23/RobustSkinWeightsTransferCode/blob/main/src/utils.py)
#   by Rinat Abdrashitov, used under the MIT License (see below).
#
# Changes were made to make the code compatible with Blender's data structures
# and to improve performance and robustness.
#
# Copyright (C) 2025 sentfromspacevr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Attribution: Developed by sentfromspacevr (https://github.com/sentfromspacevr)
#
# ---- Original MIT License Notice Follows ----
#
# The following portions of this file are based on work by Rinat Abdrashitov and are licensed under the MIT License:
#
# Copyright (c) 2024 Rinat Abdrashitov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import scipy as sp
from scipy.spatial import cKDTree
import robust_laplacian


def _dot_rows(a, b):
    return np.einsum("ij,ij->i", a, b)


def _closest_points_on_triangles(points, a, b, c):
    """Vectorized closest point on triangle for per-row inputs."""
    ab = b - a
    ac = c - a
    ap = points - a

    d1 = _dot_rows(ab, ap)
    d2 = _dot_rows(ac, ap)

    out = np.empty_like(points)
    assigned = np.zeros(points.shape[0], dtype=bool)

    mask = (d1 <= 0) & (d2 <= 0)
    out[mask] = a[mask]
    assigned |= mask

    bp = points - b
    d3 = _dot_rows(ab, bp)
    d4 = _dot_rows(ac, bp)
    mask = (~assigned) & (d3 >= 0) & (d4 <= d3)
    out[mask] = b[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    denom = d1 - d3
    v = np.divide(d1, denom, out=np.zeros_like(d1), where=np.abs(denom) > 1e-12)
    mask = (~assigned) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out[mask] = a[mask] + (v[mask, None] * ab[mask])
    assigned |= mask

    cp = points - c
    d5 = _dot_rows(ab, cp)
    d6 = _dot_rows(ac, cp)
    mask = (~assigned) & (d6 >= 0) & (d5 <= d6)
    out[mask] = c[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    denom = d2 - d6
    w = np.divide(d2, denom, out=np.zeros_like(d2), where=np.abs(denom) > 1e-12)
    mask = (~assigned) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out[mask] = a[mask] + (w[mask, None] * ac[mask])
    assigned |= mask

    va = d3 * d6 - d5 * d4
    denom = (d4 - d3) + (d5 - d6)
    w = np.divide(d4 - d3, denom, out=np.zeros_like(d4), where=np.abs(denom) > 1e-12)
    mask = (~assigned) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    out[mask] = b[mask] + (w[mask, None] * (c[mask] - b[mask]))
    assigned |= mask

    denom = va + vb + vc
    v = np.divide(vb, denom, out=np.zeros_like(vb), where=np.abs(denom) > 1e-12)
    w = np.divide(vc, denom, out=np.zeros_like(vc), where=np.abs(denom) > 1e-12)
    out[~assigned] = a[~assigned] + (ab[~assigned] * v[~assigned, None]) + (ac[~assigned] * w[~assigned, None])
    return out


def _barycentric_coordinates(points, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = _dot_rows(v0, v0)
    d01 = _dot_rows(v0, v1)
    d11 = _dot_rows(v1, v1)
    d20 = _dot_rows(v2, v0)
    d21 = _dot_rows(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = np.divide(d11 * d20 - d01 * d21, denom, out=np.zeros_like(d20), where=np.abs(denom) > 1e-12)
    w = np.divide(d00 * d21 - d01 * d20, denom, out=np.zeros_like(d20), where=np.abs(denom) > 1e-12)
    u = 1.0 - v - w
    return np.stack((u, v, w), axis=1)


def find_closest_point_on_surface(P, V, F):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """
    
    if F.shape[0] == 0:
        raise ValueError("Source mesh has no triangles")

    tri_centroids = (V[F[:, 0], :] + V[F[:, 1], :] + V[F[:, 2], :]) / 3.0
    k = min(32, F.shape[0])
    tree = cKDTree(tri_centroids)
    _, candidate_inds = tree.query(P, k=k)
    if k == 1:
        candidate_inds = candidate_inds.reshape(-1, 1)

    candidate_tris = F[candidate_inds]
    a = V[candidate_tris[:, :, 0]]
    b = V[candidate_tris[:, :, 1]]
    c = V[candidate_tris[:, :, 2]]

    p = np.repeat(P[:, None, :], k, axis=1)

    flat_p = p.reshape(-1, 3)
    flat_a = a.reshape(-1, 3)
    flat_b = b.reshape(-1, 3)
    flat_c = c.reshape(-1, 3)
    flat_cp = _closest_points_on_triangles(flat_p, flat_a, flat_b, flat_c)

    candidate_cp = flat_cp.reshape(P.shape[0], k, 3)
    candidate_sqr_d = np.sum((candidate_cp - p) ** 2, axis=2)

    best_local = np.argmin(candidate_sqr_d, axis=1)
    row_idx = np.arange(P.shape[0])
    I = candidate_inds[row_idx, best_local]
    C = candidate_cp[row_idx, best_local]
    sqrD = candidate_sqr_d[row_idx, best_local]

    F_closest = F[I, :]
    V1 = V[F_closest[:, 0], :]
    V2 = V[F_closest[:, 1], :]
    V3 = V[F_closest[:, 2], :]
    B = _barycentric_coordinates(C, V1, V2, V3)
    return sqrD, I, C, B

def interpolate_attribute_from_bary(A,B,I,F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I,:]
    a1 = A[F_closest[:,0],:]
    a2 = A[F_closest[:,1],:]
    a3 = A[F_closest[:,2],:]

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    
    A_out = a1*b1 + a2*b2 + a3*b3

    return A_out


def normalize_vec(v):
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def find_matches_closest_surface(source_verts, source_triangles, source_normals, target_verts, target_normals, source_weights, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES, flip_vertex_normal):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals
        
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals
        
        W1: #V1 by num_bones source mesh skin weights

        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold

    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
    """
    sqrD,I,C,B = find_closest_point_on_surface(target_verts,source_verts,source_triangles)
    
    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals) 
    # using the barycentric coordinates
    W2 = interpolate_attribute_from_bary(source_weights,B,I,source_triangles)
    N1_match_interpolated = interpolate_attribute_from_bary(source_normals,B,I,source_triangles)
    
    norm_N1 = np.linalg.norm(N1_match_interpolated, axis=1, keepdims=True)
    norm_N2 = np.linalg.norm(target_normals, axis=1, keepdims=True)
    normalized_N1 = np.divide(N1_match_interpolated, norm_N1, out=np.zeros_like(N1_match_interpolated), where=norm_N1 > 1e-12)
    normalized_N2 = np.divide(target_normals, norm_N2, out=np.zeros_like(target_normals), where=norm_N2 > 1e-12)

    dot_product = np.einsum('ij,ij->i', normalized_N1, normalized_N2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the dot product is in the valid range for arccos
    rad_angles = np.arccos(dot_product)
    deg_angles = np.degrees(rad_angles)
    is_distance_threshold = sqrD <= dDISTANCE_THRESHOLD_SQRD
    angle_thresholds = np.full(deg_angles.shape, dANGLE_THRESHOLD_DEGREES)

    is_deg_threshold = deg_angles <= angle_thresholds
    if flip_vertex_normal:
        deg_angles_mirror = 180 - deg_angles
        is_deg_threshold = np.logical_or(is_deg_threshold, deg_angles_mirror <= angle_thresholds)

    Matched = np.logical_and(is_distance_threshold, is_deg_threshold)    
    return Matched, W2


def inpaint(V2, F2, W2, Matched, point_cloud):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt 
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
    """
    
    if point_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(V2)
    else:
        L, M = robust_laplacian.mesh_laplacian(V2, F2)
    L = -L # igl and robust_laplacian have different laplacian conventions
    
    m_diag = M.diagonal()
    minv_diag = np.divide(1.0, m_diag, out=np.zeros_like(m_diag), where=np.abs(m_diag) > 1e-12)
    Minv = sp.sparse.diags(minv_diag)

    Q2 = -L + L*Minv*L
    Q2 = Q2.astype(np.float64)

    b = np.arange(V2.shape[0], dtype=np.int64)[Matched]
    if b.size == 0:
        return False, W2
    bc = W2[Matched, :].astype(np.float64)

    unknown_mask = np.ones(V2.shape[0], dtype=bool)
    unknown_mask[b] = False
    unknown = np.arange(V2.shape[0], dtype=np.int64)[unknown_mask]

    W_inpainted = np.zeros_like(W2, dtype=np.float64)
    W_inpainted[b, :] = bc
    result = True

    if unknown.size > 0:
        Quu = Q2[unknown[:, None], unknown]
        Qub = Q2[unknown[:, None], b]
        rhs = -(Qub @ bc)
        # Tiny diagonal regularization helps when disconnected components make Quu near-singular.
        Quu = Quu + (sp.sparse.eye(Quu.shape[0], dtype=np.float64, format="csr") * 1e-10)

        try:
            solver = sp.sparse.linalg.factorized(Quu.tocsc())
            solved = np.column_stack([solver(rhs[:, i]) for i in range(rhs.shape[1])])
        except Exception:
            try:
                solved = sp.sparse.linalg.spsolve(Quu.tocsc(), rhs)
                if solved.ndim == 1:
                    solved = solved[:, None]
            except Exception:
                return False, W2

        W_inpainted[unknown, :] = solved

    W_inpainted = W_inpainted.astype(np.float32)
    # when W2 shape = (num_verts, 1), it gets flattened to (num_verts, )
    # reshape it back to initial shape, limit_mask expects 2d array
    if result:
        W_inpainted = W_inpainted.reshape(W2.shape)
    return result, W_inpainted # TODO: Add results
    
    
def limit_mask(weights, adjacency_matrix, dilation_repeat=5, limit_num=4):
    if weights.shape[1] <= limit_num: return np.zeros_like(weights)
    
    count = np.count_nonzero(weights, axis=1)
    to_limit = count > limit_num
    k = weights.shape[1] - limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sp.sparse.csr_array(erode_mask).astype(np.float32)
    adj_mat = adjacency_matrix
    degrees = np.asarray(adj_mat.sum(axis=1)).reshape(-1)
    inv_degrees = np.divide(1.0, degrees, out=np.zeros_like(degrees, dtype=np.float32), where=degrees > 0)
    smooth_mat = sp.sparse.diags(inv_degrees) @ adj_mat
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)
    
    return erode_mask.toarray()


def smooth_weigths(verts, weights, matched, adjacency_matrix, adjacency_list, num_smooth_iter_steps, smooth_alpha, distance_threshold):
    not_matched = ~matched
    VIDs_to_smooth = np.zeros(verts.shape[0], dtype=bool)

    def get_points_within_distance(V, VID, distance=distance_threshold):
        """
        Get all neighbours of vertex VID within dDISTANCE_THRESHOLD
        """
        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            if vv < len(adjacency_list):
                neigh = adjacency_list[vv]
                for nn in neigh:
                    if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID,:]-V[nn]) < distance:
                        VIDs_to_smooth[nn] = True
                        if nn not in queue:
                            queue.append(nn)

    for i in range(verts.shape[0]):
        if not_matched[i]:
            get_points_within_distance(verts, i, distance_threshold)
            
    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = np.asarray(adj_mat.sum(axis=1)).reshape(-1)
    inv_degrees = np.divide(1.0, degrees, out=np.zeros_like(degrees, dtype=np.float32), where=degrees > 0)
    smooth_mat = sp.sparse.diags(inv_degrees) @ adj_mat
    weights_smoothed = sp.sparse.csr_array(weights)
    for _ in range(num_smooth_iter_steps):
        weights_smoothed = (1 - smooth_alpha) * weights_smoothed + smooth_alpha * (smooth_mat @ weights_smoothed)
        weights_smoothed[~VIDs_to_smooth] = weights[~VIDs_to_smooth]
    return np.asarray(weights_smoothed.todense(), dtype=np.float32)
            
