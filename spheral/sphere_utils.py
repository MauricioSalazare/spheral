import warnings

import numpy as np
from scipy.optimize import leastsq
import pandas as pd

from spheral.data_models import CartesianCoord, SphereParameters


def sorted_eignevalues(S):
    # Eigenvalues and right eigenvectors of the covariance matrix (square array)
    # eigenvalues, eigenvectors = np.linalg.eig(S)
    eigenvalues, eigenvectors = np.linalg.eigh(S)  # Assumes hermitian symmetric matrix

    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx_sorted]
    eigenvectors_sorted = eigenvectors[
        :, idx_sorted
    ]  # The eigenvectors are along columns (different from SVD method)

    return eigenvalues_sorted, eigenvectors_sorted


def fitfunc(p, coords):
    x0, y0, z0, _ = p
    x, y, z = coords.T
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)



def fit_sphere(
    z_frame_full: pd.DataFrame, wireframe_res: int = 100
) -> tuple[pd.DataFrame, SphereParameters]:

    coords_pca = z_frame_full.iloc[
        :, :3
    ].values  # Take 3 principal components by default
    # p0 = [x0, y0, z0, R]
    p0 = [0, 0, 0, 1]

    errfunc = lambda p, x: fitfunc(p, x) - p[3]
    p1, flag = leastsq(errfunc, p0, args=(coords_pca,))

    if flag == 0:
        raise ValueError("Sphere fitting failed. Least squares did not converge.")

    # Shift the data around the origin, the offset is found with the spherical fit
    z_frame_centered = z_frame_full.copy()
    offset = np.array([p1[0], p1[1], p1[2]])
    z_frame_centered.iloc[:, :3] = z_frame_centered.iloc[:, :3] - offset

    sphere_parameters = SphereParameters(
        center=CartesianCoord(x=p1[0], y=p1[1], z=p1[2]),
        radius=p1[3],
        resolution=wireframe_res,
    )

    return z_frame_centered, sphere_parameters


def random_VMF(mu, kappa, size=None):
    """
    Von Mises-Fisher distribution sampler with mean direction mu and concentration kappa.
    Source: https://hal.science/hal-04004568
    """

    # parse input parameters

    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))

    mu = np.asarray(mu)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape

    # z component: radial samples perpendicular to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    # sample angles (in cos and sin form)
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos**2)

    # Combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]

    return x.reshape((*shape, d))


def _random_VMF_cos(d: int, kappa: float, n: int):
    """
    Generate n iid samples t with density function given by
        p(t) = someConstant * (1 - t**2)**((d-2)/2) * exp(kappa * t)
    """

    # b = Eq. 4 of https://doi.org/10.1080/03610919408813161
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -np.random.exponential(size=m)
        out.append(t[accept])
        found += len(out[-1])

    return np.concatenate(out)[:n]


def wrap_angle_radians(angle_radians):
    """
    This function ensures that angles are wrapped to the range [-np.pi, np.pi].

    phi (azimuthal angle): Measured in the XY-plane from the positive X-axis, wrapping occurs naturally
    within [-np.pi, np.pi]
    """
    return (angle_radians + np.pi) % (2 * np.pi) - np.pi


def mean_angle_radians(angle_radians):
    """This computes the mean of a set of angles in radians using vector summation"""
    return np.arctan2(np.sin(angle_radians).sum(), np.cos(angle_radians).sum())


def shift_by_angle_mean(angle_degrees):
    """
    This shifts the input angles (in degrees) so that they are centered around their mean, and wraps the result back
    to the range [-np.pi, np.pi]
    """
    angle_mean_rad = mean_angle_radians(np.deg2rad(angle_degrees))
    angle_centered_degrees = np.rad2deg(
        wrap_angle_radians(np.deg2rad(angle_degrees) - angle_mean_rad)
    )

    return angle_centered_degrees


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically, for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    # scale is an array
    else:
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = np.asarray(scale, copy=True)
        scale[constant_mask] = 1.0
        return scale


def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True, ddof=0):
    """
    Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    ddof : int, default=0
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number
        of non-NaN elements. By default ddof is zero.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.



    """

    X = np.asarray(X, copy=copy).astype(float)

    if with_mean:
        mean_ = np.nanmean(X, axis)
    if with_std:
        scale_ = np.nanstd(X, axis, ddof=ddof)
    # Xr is a view on the original array that enables easy use of
    # broadcasting on the axis in which we are interested in
    # Any change in X is a change in Xr
    Xr = np.rollaxis(X, axis)

    if with_mean:
        Xr -= mean_
        mean_1 = np.nanmean(Xr, axis=0)
        # Verify that mean_1 is 'close to zero'. If X contains very
        # large values, mean_1 can also be very large, due to a lack of
        # precision of mean_. In this case, a pre-scaling of the
        # concerned feature is efficient, for instance by its mean or
        # maximum.
        if not np.allclose(mean_1, 0):
            warnings.warn(
                "Numerical issues were encountered "
                "when centering the data "
                "and might not be solved. Dataset may "
                "contain too large values. You may need "
                "to prescale your features."
            )
            Xr -= mean_1
    if with_std:
        scale_ = _handle_zeros_in_scale(scale_, copy=False)
        Xr /= scale_
        if with_mean:
            mean_2 = np.nanmean(Xr, axis=0)
            # If mean_2 is not 'close to zero', it comes from the fact that
            # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
            # if mean_1 was close to zero. The problem is thus essentially
            # due to the lack of precision of mean_. A solution is then to
            # subtract the mean again:
            if not np.allclose(mean_2, 0):
                warnings.warn(
                    "Numerical issues were encountered "
                    "when scaling the data "
                    "and might not be solved. The standard "
                    "deviation of the data is probably "
                    "very close to 0. "
                )
                Xr -= mean_2
    return X
