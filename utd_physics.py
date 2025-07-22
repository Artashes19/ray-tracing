import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------
TWOPI = 2 * np.pi


def _norm(a):
    """Normalize angle to [0, 2π)."""
    return a % TWOPI


def _angle_between(a, b):
    """Smallest absolute angle between two directions in [0, π]."""
    return abs((_norm(a - b + np.pi)) - np.pi)


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------

def calculate_utd_coefficient(source, edge, observer, wall_points, frequency):
    """Return (D_GTD_complex, phi0_vis, phi_vis, face_ref_angle).

    * phi0_vis  – smallest angle between 0-face and incident ray (deg)
    * phi_vis   – clockwise angle from 0-face to diffracted ray (deg)
    * face_ref_angle returned in radians for diagnostics
    """

    # --- basic geometry vectors ---
    p1, p2 = wall_points
    wall_vec = np.array(p2) - np.array(p1)
    if np.allclose(edge, p2):
        wall_vec = -wall_vec  # ensure vector points away from edge into the wall

    wall_angle = _norm(np.arctan2(wall_vec[1], wall_vec[0]))

    # Directions for incident (edge->source) and diffracted (edge->observer)
    inc_dir = _norm(np.arctan2(source[1] - edge[1], source[0] - edge[0]))
    diff_dir = _norm(np.arctan2(observer[1] - edge[1], observer[0] - edge[0]))

    # Clockwise angles from wall to rays
    phi0 = _norm(wall_angle - inc_dir)  # incident
    phi  = _norm(wall_angle - diff_dir) # diffracted

    phi0_deg = np.degrees(phi0)
    phi_deg  = np.degrees(phi)

    # ---------------------------------------------------------
    # GTD diffraction coefficient (half-plane)
    # ---------------------------------------------------------
    wavelength = SPEED_OF_LIGHT / frequency
    k = TWOPI / wavelength
    n = 1  # half-plane

    # Formula: 1/(2√(2πk)) [ -sec((φ-φ0)/2) + sec((φ+φ0)/2) ]
    sec_minus = 1 / np.cos((phi - phi0) / 2)
    sec_plus  = 1 / np.cos((phi + phi0) / 2)
    D_gtd = (-sec_minus + sec_plus) / (2 * np.sqrt(2 * np.pi * k))

    return complex(D_gtd), np.radians(phi0_deg), np.radians(phi_deg), wall_angle

if __name__ == '__main__':
    print("--- Running Basic Test Case ---")
    source_pos = (120, 100)
    wall_geom = ((100, 120), (50, 120))
    diffracting_edge = wall_geom[0]
    observer_pos = (80, 180)
    freq = 900e6
    
    calculate_utd_coefficient(source_pos, diffracting_edge, observer_pos, wall_geom, freq)
    print("\n--- Test Case Complete ---") 