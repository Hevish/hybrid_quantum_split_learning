"""
Reconstruction-attack evaluation metrics.

Provides masked MSE, cosine distance, and DSSIM between original and
reconstructed images, following the methodology in the HQSL paper.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def _create_mask(img: np.ndarray, rec: np.ndarray):
    """Return the non-zero masked regions of *img* and *rec*."""
    assert img.shape == rec.shape, "Images must have the same dimensions"
    nz = np.where((img > 0) | (rec > 0))
    return img[nz], rec[nz]


def cosine_distance(img: np.ndarray, rec: np.ndarray) -> float:
    a, b = _create_mask(img, rec)
    a, b = np.squeeze(a), np.squeeze(b)
    sim = cosine_similarity(
        a.flatten().reshape(1, -1), b.flatten().reshape(1, -1)
    ).item()
    return 1.0 - sim


def masked_mse(img: np.ndarray, rec: np.ndarray) -> float:
    a, b = _create_mask(img, rec)
    return float(mean_squared_error(a, b))


def dssim(img: np.ndarray, rec: np.ndarray) -> float:
    """Structural dissimilarity (1 - normalised SSIM)."""
    a, b = _create_mask(img, rec)
    s = ssim(a.flatten().reshape(1, -1),
             b.flatten().reshape(1, -1),
             channel_axis=0, data_range=2.0)
    s = max(s, 0.0)
    return 1.0 - (1.0 + s) / 2.0


def log_spectral_distance(img: np.ndarray, rec: np.ndarray,
                          floor: float = 1e-10) -> float:
    """Log-spectral distance between two spectrograms.

    Floors values to avoid log(0) before computing RMS of
    log-domain differences.
    """
    a = np.maximum(img, floor)
    b = np.maximum(rec, floor)
    sq_diff = (np.log10(a) - np.log10(b)) ** 2
    return float(np.sqrt(np.nanmean(sq_diff)))


def compute_metrics_batch(images, reconstructions, *, include_lsd: bool = False):
    """Compute mean & std of all metrics over a batch.

    Parameters
    ----------
    images, reconstructions : array-like
        Batches of NumPy arrays (typically ``tensor.cpu().numpy()``).
    include_lsd : bool
        If *True* also compute Log Spectral Distance (audio datasets).

    Returns
    -------
    dict  with keys ``mse``, ``cos_dist``, ``dssim``, (and optionally
    ``lsd``), each mapping to ``(mean, std)``.
    """
    mses, coss, dss, lsds = [], [], [], []
    for img, rec in zip(images, reconstructions):
        mses.append(masked_mse(img, rec))
        coss.append(cosine_distance(img, rec))
        dss.append(dssim(img, rec))
        if include_lsd:
            lsds.append(log_spectral_distance(img, rec))
    mses, coss, dss = map(np.array, (mses, coss, dss))
    result = {
        "mse": (mses.mean(), mses.std()),
        "cos_dist": (coss.mean(), coss.std()),
        "dssim": (dss.mean(), dss.std()),
    }
    if include_lsd:
        lsds = np.array(lsds)
        result["lsd"] = (lsds.mean(), lsds.std())
    return result
