from typing import Optional, List, Any
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, vonmises, skewnorm

from spheral.data_models import (
    SphereParameters,
    EigenProfiles,
    PolarMean,
    OutlierBound,
    OutlierModel,
    OutlierParameters,
    SphereOutlierModel,
    BoundType, polar_coordinates,
)
from spheral.sphere_utils import (
    sorted_eignevalues,
    fit_sphere,
    shift_by_angle_mean,
    mean_angle_radians,
    scale,
)


class Sphere:
    """
    Spherical representation for the dataset
    """

    def __init__(self, id_sphere: str = None):
        self.id_sphere = id_sphere
        self.rlps: pd.DataFrame | None = None
        self.rlps_scaled = None
        self.explainability = None
        self.z_frame = None
        # self.sphere_surface = None
        self.sphere_parameters: SphereParameters | None = None
        self.polar_coord = None
        self.outlier_model = None
        self.outliers_frame = None
        self.eigen_profiles: Optional[EigenProfiles] = None

    def fit(self, data: np.ndarray | pd.DataFrame, exclude_ids: list | None = None):
        """
        Process the representative load profiles

        Parameters:
        -----------
            data: np.ndarray|pd.DataFrame:  it has 2-dimensions (n_samples[rows], n_features[columns])
                The dataset is the representative load profiles (RLPs) of a month.
                Each row is a RLP, and the index (pd.DataFrame) is the DALIBOXID.
            exclude_ids: list[str]
                List of ids that should be omitted in the computation of the sphere and outliers.

        """
        if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
            raise TypeError("Data must be an ndarray or a pd.DataFrame")

        if isinstance(data, np.ndarray):
            rlps_frame = pd.DataFrame(data)  # Adds automatic indexing to the samples
        else:
            rlps_frame = data

        if exclude_ids is not None:
            if isinstance(exclude_ids, list):
                rlps_frame = rlps_frame[~rlps_frame.index.isin(exclude_ids)]
            else:
                raise ValueError("Exclude ids must be list or None")

        # Standardize load profiles
        X = scale(
            rlps_frame, axis=1
        )  # (n_samples, n_features)  # axis=1 standardize each sample (row mean)
        rlps_frame_scaled = pd.DataFrame(X, index=rlps_frame.index)

        # Normalize
        n_samples, n_time_steps = X.shape
        X = X / np.sqrt(n_time_steps)

        # Center the data among features for PCA
        X_centered = X - X.mean(axis=0)

        # Compute PCA and z-scores
        S = (X_centered.transpose() @ X_centered) / (n_samples - 1)  # Covariance matrix
        eigenvalues_sorted, eigenvectors_sorted = sorted_eignevalues(
            S
        )  # eigen_vectors = columns of the array
        z_full = X_centered @ eigenvectors_sorted

        # Principal components frame
        z_frame_full = pd.DataFrame(
            z_full, columns=["PC" + str(ii) for ii in range(z_full.shape[1])]
        )
        z_frame_full.index = rlps_frame.index

        # Explainability of the components
        explainability = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted)

        # Fit the sphere
        z_frame_centered, sphere_parameters = fit_sphere(z_frame_full)

        (r_pca, phi_pca_degrees, theta_pca_degrees) = polar_coordinates(
            z_frame_centered[["PC0", "PC1", "PC2"]].values
        )

        phi_angle_deg_centered = shift_by_angle_mean(angle_degrees=phi_pca_degrees)
        theta_angle_deg_centered = shift_by_angle_mean(angle_degrees=theta_pca_degrees)

        polar_mean = PolarMean(
            radius=sphere_parameters.radius,
            theta_mean_deg=np.rad2deg(
                mean_angle_radians(np.deg2rad(theta_pca_degrees))
            ),
            phi_mean_deg=np.rad2deg(mean_angle_radians(np.deg2rad(phi_pca_degrees))),
        )

        polar_coord = pd.DataFrame(
            index=z_frame_centered.index,
            data={
                "r": r_pca,
                "phi": phi_pca_degrees,
                "theta": theta_pca_degrees,
                "phi_centered": phi_angle_deg_centered,
                "theta_centered": theta_angle_deg_centered,
            },
        )
        self.eigen_profiles = EigenProfiles(
            eig1=eigenvectors_sorted[:, 0],
            eig2=eigenvectors_sorted[:, 1],
            eig3=eigenvectors_sorted[:, 2],
        )

        self.rlps = rlps_frame
        self.rlps_scaled = rlps_frame_scaled

        self.excluded_ids = exclude_ids

        self.dali_boxes_sphere = sorted(self.rlps.index.to_list())

        self.explainability = explainability

        self.z_frame = z_frame_centered
        # self.sphere_surface = sphere_surface
        self.sphere_parameters = sphere_parameters

        self.polar_coord = polar_coord
        self.polar_mean = polar_mean

    def _process_parametric_outliers(
        self,
        name: str,
        values: np.ndarray,
        param_bounds: OutlierBound,
        distribution: rv_continuous,
    ):
        """Compute parametric outliers using a distribution."""
        if name in ["phi", "theta"] and distribution == vonmises:
            # Compute vonmises parameters only for phi and theta
            # First two parameters [:2] are kappa_angle, mean_angle, _
            params = vonmises.fit(np.deg2rad(values), fscale=1)[:2]
            lower_bound, upper_bound = np.rad2deg(
                distribution.ppf([param_bounds.lb, param_bounds.ub], *params)
            )
        elif name == "radius":
            # alpha_, mean_skew, scale_skew = skewnorm_parameters
            params = distribution.fit(values)
            lower_bound, upper_bound = distribution.ppf(
                [param_bounds.lb, param_bounds.ub], *params
            )
        else:
            raise ValueError(
                f"Unknown dimension {name}. It muste be either 'radius', 'phi' or 'theta'."
            )

        return OutlierModel(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            distribution=distribution,
            parameters=params,
        )

    def _process_non_parametric_outliers(
        self, values: np.ndarray, param_bounds: OutlierBound
    ):
        """Compute non-parametric outliers using IQR."""
        lower_bound, upper_bound = param_bounds.iqr_bounds(values)
        return OutlierModel(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            distribution=None,
            parameters=None,
        )

    def _compute_outlier_model(
        self,
        name: str,
        values: np.ndarray,
        param_bounds: OutlierBound,
        distribution: rv_continuous = None,
    ):
        """Generic handler for computing outlier models."""
        if param_bounds.bound_type == BoundType.PARAMETRIC:
            if not isinstance(distribution, rv_continuous):
                raise TypeError(
                    "Distribution must be an instance of rv_continuous (Scipy stats probability models)."
                )
            return self._process_parametric_outliers(
                name, values, param_bounds, distribution
            )
        elif param_bounds.bound_type == BoundType.NON_PARAMETRIC:
            return self._process_non_parametric_outliers(values, param_bounds)
        else:
            raise ValueError(
                f"Invalid bound type for {name}: {param_bounds.bound_type}"
            )

    def compute_outliers(
        self, outliers_parameters: OutlierParameters = OutlierParameters.default()
    ):
        # Extract and preprocess values
        r_pca = self.polar_coord["r"].values
        phi_angle_deg_centered = self.polar_coord["phi_centered"].values
        theta_angle_deg_centered = self.polar_coord["theta_centered"].values

        # Process radius
        radius_result = self._compute_outlier_model(
            "radius", r_pca, outliers_parameters.radius, distribution=skewnorm
        )

        # Process phi
        phi_result = self._compute_outlier_model(
            "phi",
            phi_angle_deg_centered,
            outliers_parameters.phi,
            distribution=vonmises,
        )

        # Process theta
        theta_result = self._compute_outlier_model(
            "theta",
            theta_angle_deg_centered,
            outliers_parameters.theta,
            distribution=vonmises,
        )

        # Compute the outliers outside the boundaries
        idx_radius = r_pca < radius_result.lower_bound
        idx_phi = (phi_angle_deg_centered < phi_result.lower_bound) | (
            phi_angle_deg_centered > phi_result.upper_bound
        )
        idx_theta = (theta_angle_deg_centered < theta_result.lower_bound) | (
            theta_angle_deg_centered > theta_result.upper_bound
        )

        idx_outliers_by_angle = idx_phi | idx_theta
        idx_outliers = idx_radius | idx_outliers_by_angle

        outliers_frame = pd.DataFrame(
            {
                "sample_id": self.polar_coord.index,
                "is_outlier": idx_outliers,
                "by_radius": idx_radius,
                "by_phi": idx_phi,
                "by_theta": idx_theta,
                "by_angle": idx_outliers_by_angle,
                "radius": r_pca,
                "phi_degrees": self.polar_coord["phi"].values,
                "theta_degrees": self.polar_coord["theta"].values,
                "phi_centered_degrees": phi_angle_deg_centered,
                "theta_centered_degrees": theta_angle_deg_centered,
            }
        )

        self.sphere_outlier_model = SphereOutlierModel(
            radius=radius_result, phi=phi_result, theta=theta_result
        )
        self.outliers_frame = outliers_frame

        return outliers_frame

    def get_outliers_ids(self, by: str = "radius") -> List[Any]:
        if self.outliers_frame is None:
            raise ValueError(
                "Outliers frame is empty. run 'compute_outliers()' method first."
            )

        option_mapping = {
            "all": "is_outlier",
            "radius": "by_radius",
            "phi": "by_phi",
            "theta": "by_theta",
            "angle": "by_angle",
        }

        if by not in option_mapping.keys():
            raise ValueError(
                f"Invalid value for {by}. Valid options are {option_mapping.keys()}"
            )

        return self.outliers_frame["sample_id"][
            self.outliers_frame[option_mapping[by]].values
        ].to_list()
