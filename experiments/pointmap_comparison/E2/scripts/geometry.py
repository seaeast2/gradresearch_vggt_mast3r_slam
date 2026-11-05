"""Similarity alignment primitives using column-vector Sim(3) convention."""
import numpy as np


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    return points @ m[:3, :3].T + m[:3, 3]


def estimate_sim3(source, target):
    """Return 4x4 S mapping source points onto target points (Umeyama, no reflection)."""
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("source/target must be matching Nx3 arrays with N >= 3")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Sim(3) inputs must be finite")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    variance = np.sum(xc * xc) / len(x)
    if variance <= 1e-14:
        raise ValueError("source points have zero spatial variance")
    covariance = (yc.T @ xc) / len(x)
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("estimated Sim(3) scale is not positive and finite")
    translation = my - scale * rotation @ mx
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def prefix_degeneracy(centers, minimum=60, min_displacement_m=0.5, min_lambda2_ratio=0.01):
    c = np.asarray(centers, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("camera centers must have shape Nx3")
    if len(c) < minimum:
        return {"status": "insufficient_prefix", "count": len(c), "minimum": minimum}
    if not np.isfinite(c).all():
        return {"status": "non_finite_centers", "count": len(c)}
    displacement = float(np.linalg.norm(c - c[0], axis=1).max())
    eigen = np.linalg.eigvalsh(np.cov(c.T))[::-1]
    ratio = float(eigen[1] / eigen[0]) if eigen[0] > 0 else 0.0
    reasons = []
    if displacement < min_displacement_m:
        reasons.append("insufficient_translation_span")
    if ratio < min_lambda2_ratio:
        reasons.append("near_collinear_trajectory")
    return {"status": "degenerate" if reasons else "passed", "count": len(c),
            "max_displacement_m": displacement, "center_covariance_eigenvalues_m2": eigen.tolist(),
            "lambda2_over_lambda1": ratio, "reasons": reasons}


def estimate_prefix_sim3(source_centers, target_centers, minimum=60):
    check = prefix_degeneracy(source_centers, minimum=minimum)
    if check["status"] != "passed":
        return {"status": check["status"], "diagnostics": check}
    try:
        matrix = estimate_sim3(source_centers, target_centers)
    except ValueError as exc:
        return {"status": "alignment_failed", "diagnostics": check, "error": str(exc)}
    return {"status": "passed", "diagnostics": check, "matrix_source_to_target": matrix.tolist()}
