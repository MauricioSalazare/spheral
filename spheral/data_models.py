from dataclasses import dataclass
from enum import Enum
from typing import Optional, Iterator, Tuple, Any

import numpy as np


@dataclass
class EigenProfiles:
    eig1: Optional[np.ndarray] = None
    eig2: Optional[np.ndarray] = None
    eig3: Optional[np.ndarray] = None

    def __repr__(self):
        """
        Override the default string representation to show whether each attribute is filled.
        """
        attributes = {
            "eig1": self.eig1 is not None,
            "eig2": self.eig2 is not None,
            "eig3": self.eig3 is not None,
        }
        filled_status = ", ".join(
            [
                f"{key}: {'Filled' if value else 'Empty'}"
                for key, value in attributes.items()
            ]
        )
        return f"<EigenProfiles({filled_status})>"

    def __iter__(self) -> Iterator[Optional[np.ndarray]]:
        """
        Make the class iterable by yielding each eigen profile.
        """
        for attr in ("eig1", "eig2", "eig3"):
            yield getattr(self, attr)


@dataclass
class CartesianCoord:
    x: float
    y: float
    z: float

    def to_polar(self, angle_units="degrees"):
        """
        Converts the cartesian coordinates to polar coordinates.

        Parameter:
        ----------
            angle_units: str: Either 'degrees' or 'radians'.

        Return:
        -------
            Cartesian coordinates in polar coordinates.
        """
        if angle_units == "degrees":
            return polar_coordinates(np.atleast_2d([self.x, self.y, self.z]))
        elif angle_units == "radians":
            return np.deg2rad(
                polar_coordinates(np.atleast_2d([self.x, self.y, self.z]))
            )
        else:
            raise ValueError(f'angle_units must be either "degrees" or "radians"')


@dataclass
class SphereSurface:
    """
    Surface mesh data for a sphere with the specified radius and resolution.
    """

    radius: float
    resolution: int
    theta_angles: np.ndarray = None
    phi_angles: np.ndarray = None
    x_mesh: np.ndarray = None
    y_mesh: np.ndarray = None
    z_mesh: np.ndarray = None

    def __post_init__(self):
        # Generate data for the sphere wireframe
        self.theta_angles = np.linspace(0, np.pi, self.resolution)  # Azimuthal angle
        self.phi_angles = np.linspace(0, 2 * np.pi, self.resolution)  # Polar angle

        # Create grid for wireframe
        theta, phi = np.meshgrid(self.theta_angles, self.phi_angles)
        self.x_mesh = self.radius * np.sin(theta) * np.cos(phi)
        self.y_mesh = self.radius * np.sin(theta) * np.sin(phi)
        self.z_mesh = self.radius * np.cos(theta)

    def __repr__(self):
        # Check if optional fields are not None and display True or None
        optional_fields = {
            "theta_angles": self.theta_angles is not None,
            "phi_angles": self.phi_angles is not None,
            "x_mesh": self.x_mesh is not None,
            "y_mesh": self.y_mesh is not None,
            "z_mesh": self.z_mesh is not None,
        }
        # Format the output
        optional_str = ", ".join(
            f"{key}=True" for key, value in optional_fields.items() if value
        )
        return f"SphereSurface(radius={self.radius}, resolution={self.resolution}, {optional_str})"


@dataclass
class SphereParameters:
    center: CartesianCoord
    radius: float
    resolution: int = 100
    surface: Optional[SphereSurface] = None

    def __post_init__(self):
        self.surface = SphereSurface(self.radius, self.resolution)

    def __repr__(self):
        return (
            f"SphereParameters(center={self.center}, "
            f"radius={self.radius}, "
            f"resolution={self.resolution}, "
            f"surface={True if self.surface is not None else False})"
        )


@dataclass
class PolarMean:
    radius: float
    theta_mean_deg: float
    phi_mean_deg: float
    theta_mean_rad: Optional[float] = None
    phi_mean_rad: Optional[float] = None
    cartesian: Optional[CartesianCoord] = None

    def __post_init__(self):
        self.theta_mean_rad = np.deg2rad(self.theta_mean_deg)
        self.phi_mean_rad = np.deg2rad(self.phi_mean_deg)
        vector = spherical_to_cartesian(
            theta=self.theta_mean_rad, phi=self.phi_mean_rad, r=self.radius
        )
        self.cartesian = CartesianCoord(*vector)

    def __repr__(self):
        formatted_attributes = []
        for key, value in vars(self).items():
            if isinstance(value, float):
                formatted_attributes.append(f"{key}={value:.4f}")
            else:
                formatted_attributes.append(f"{key}={value}")
        return f"PolarMean({', '.join(formatted_attributes)})"


class BoundType(Enum):
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non-parametric"


@dataclass
class OutlierBound:
    lb: float  # Lower bound or coefficient (e.g., constant for Q1 - lb * IQR in non-parametric mode)
    ub: float  # Upper bound or coefficient (e.g., constatn for Q3 + ub * IQR in non-parametric mode)
    bound_type: BoundType  # Specifies if the bound is 'parametric' or 'non-parametric'

    def __post_init__(self):
        """Validate the initialization values."""
        if self.bound_type == BoundType.PARAMETRIC:
            if self.lb < 0 or self.ub > 1:
                raise ValueError(
                    "For parametric bounds, lb and ub must be within [0, 1]."
                )
        elif self.bound_type == BoundType.NON_PARAMETRIC:
            if self.lb < 0 or self.ub < 0:
                raise ValueError(
                    "For non-parametric bounds, lb and ub must be non-negative."
                )

    def iqr_bounds(self, values: np.ndarray) -> Tuple[float, float]:
        if self.bound_type == BoundType.PARAMETRIC:
            raise UserWarning(
                "The bounds are originally set as parametric. Calculating non-parametric bounds."
            )

        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.lb * iqr
        upper_bound = q3 + self.ub * iqr

        return lower_bound, upper_bound


@dataclass
class OutlierParameters:
    radius: OutlierBound
    phi: OutlierBound
    theta: OutlierBound

    def set_bounds(self, parameter: str, lb: float, ub: float) -> None:
        if hasattr(self, parameter):
            bound = getattr(self, parameter)
            bound.lb = lb
            bound.ub = ub
        else:
            raise AttributeError(
                f"Parameter '{parameter}' does not exist. Options are 'radius', 'phi', 'theta'."
            )

    @staticmethod
    def default() -> "OutlierParameters":
        """Return the default outlier parameters."""
        return OutlierParameters(
            radius=OutlierBound(
                lb=3.0, ub=1.5, bound_type=BoundType.NON_PARAMETRIC
            ),  # Extreme outlier
            phi=OutlierBound(lb=0.03, ub=0.97, bound_type=BoundType.PARAMETRIC),  # 97%
            theta=OutlierBound(
                lb=0.005, ub=0.995, bound_type=BoundType.PARAMETRIC
            ),  # 95%
        )


@dataclass
class OutlierModel:
    lower_bound: float  # Value in actual units, e.g., degrees, radius units.
    upper_bound: float  # Value in actual units, e.g., degrees, radius units.
    distribution: Any  # The distribution class (e.g., scipy.stats.skewnorm or vonmises)
    parameters: Tuple[Any, ...] | None  # Parameters of the distribution class


@dataclass
class SphereOutlierModel:
    radius: OutlierModel
    phi: OutlierModel
    theta: OutlierModel


def spherical_to_cartesian(theta, phi, r: float = 1):
    """
    Convert spherical coordinates to Cartesian coordinates.

    :param theta: Polar angle in radians (angle from the positive Z-axis)
    :param phi: Azimuthal angle in radians (angle in the XY-plane from the X-axis)
    :param r: Radius (default is 1)
    :return: x, y, z Cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def polar_coordinates(coords_pca):
    """
    Polar coordinates are referenced as ISO.
        R: radius from the center
        theta: Angle from the z+ axis
        phi: Azimuth from the x+ axis


    This is the correct version as guided by wikipedia, real data shows that it is also correct

    Input:
    ------
        coords_pca: np.array: dim = (samples, 3). Each column corresponds to one dimension:
            e.g. columns == z_0, z_1, z_2, which are x, y, z coordinates.

    Return:
    -------
        tuple = (r, theta, phi) and angles are in degrees

    """

    r_pca = np.sqrt(
        coords_pca[:, 0] ** 2 + coords_pca[:, 1] ** 2 + coords_pca[:, 2] ** 2
    )

    # -------------------------------------------------------------------------------------
    ## PHI CONVERSION (AZIMUTHAL ANGLE)
    # phi_pca = np.arctan(coords_pca[:, 1] / coords_pca[:, 0])
    # phi_pca_degrees = np.degrees(phi_pca)

    phi_pca_degrees = np.zeros(coords_pca.shape[0])

    # Rules of quadrants for azimuthal angle.
    # Six rules for the phi angle: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # Section Coordinates system conversions -> Cartesian coordinates

    idx_rule_1 = coords_pca[:, 0] > 0.0  # Do nothing
    idx_rule_2 = (coords_pca[:, 0] < 0.0) & (coords_pca[:, 1] >= 0.0)
    idx_rule_3 = (coords_pca[:, 0] < 0.0) & (coords_pca[:, 1] < 0.0)
    idx_rule_4 = (coords_pca[:, 0] == 0.0) & (coords_pca[:, 1] > 0.0)
    idx_rule_5 = (coords_pca[:, 0] == 0.0) & (coords_pca[:, 1] < 0.0)
    idx_rule_6 = (coords_pca[:, 0] == 0.0) & (coords_pca[:, 1] == 0.0)

    # phi_pca_degrees[idx_rule_2] += 180.0
    # phi_pca_degrees[idx_rule_3] -= 180.0
    # phi_pca_degrees[idx_rule_4] = 90.0
    # phi_pca_degrees[idx_rule_5] = -90.0
    # phi_pca_degrees[idx_rule_6] = np.nan

    phi_pca_degrees[idx_rule_1] = np.degrees(
        np.arctan(coords_pca[idx_rule_1, 1] / coords_pca[idx_rule_1, 0])
    )
    phi_pca_degrees[idx_rule_2] = (
        np.degrees(np.arctan(coords_pca[idx_rule_2, 1] / coords_pca[idx_rule_2, 0]))
        + 180.0
    )
    phi_pca_degrees[idx_rule_3] = (
        np.degrees(np.arctan(coords_pca[idx_rule_3, 1] / coords_pca[idx_rule_3, 0]))
        - 180.0
    )
    phi_pca_degrees[idx_rule_4] = 90.0
    phi_pca_degrees[idx_rule_5] = -90.0
    phi_pca_degrees[idx_rule_6] = np.nan

    # phi_pca_degrees = np.degrees(np.sign(coords_pca[:, 1]) * np.arccos(coords_pca[:, 0] / (np.sqrt(coords_pca[:, 0] ** 2 + coords_pca[:, 1] ** 2))))

    # idx_quadrant_3_4 = coords_pca[:, 0] < 0.0
    # idx_quadrant_2 = (coords_pca[:, 0] > 0.0) & (coords_pca[:, 1] < 0.0)

    # Change the reference (The positive X axis is the reference point)
    # phi_pca_degrees[idx_quadrant_2] += 360.0
    # phi_pca_degrees[idx_quadrant_3_4] += 180.0

    # -------------------------------------------------------------------------------------
    ## THETA CONVERSION (POLAR ANGLE)
    theta_pca = np.arctan(
        np.sqrt((coords_pca[:, 0] ** 2 + coords_pca[:, 1] ** 2)) / coords_pca[:, 2]
    )
    theta_pca_degrees = np.degrees(theta_pca)

    # Rules of quadrants for theta angle
    # idx_rule_1 = coords_pca[:, 2] > 0.0  # Do nothing
    idx_rule_2 = coords_pca[:, 2] < 0.0
    idx_rule_3 = (coords_pca[:, 2] == 0.0) & (
        coords_pca[:, 0] * coords_pca[:, 1] != 0.0
    )
    idx_rule_4 = (
        (coords_pca[:, 0] == 0.0)
        & (coords_pca[:, 1] == 0.0)
        & (coords_pca[:, 2] == 0.0)
    )

    theta_pca_degrees[idx_rule_2] += 180.0
    theta_pca_degrees[idx_rule_3] = 90.0
    theta_pca_degrees[idx_rule_4] = np.nan

    # # Rules for the polar angle
    # idx_quadrant_5_6_7_8 = coords_pca[:, 2] < 0.0
    #
    # # Change the reference (The positive Z axis is the reference point)
    # theta_pca_degrees[idx_quadrant_5_6_7_8] += 180

    return (r_pca, phi_pca_degrees, theta_pca_degrees)
