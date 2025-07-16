import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

def calculate_utd_coefficient(source, edge, observer, wall_points, frequency):
    """
    Calculates the Uniform Theory of Diffraction (UTD) coefficient.

    This function will compute the coefficient D based on the geometry of the
    source, edge, and observer, and the frequency of the signal.

    Args:
        source (tuple): (x, y) coordinates of the transmitter.
        edge (tuple): (x, y) coordinates of the diffracting edge.
        observer (tuple): (x, y) coordinates of the observation point.
        wall_points (tuple): A tuple of two (x, y) tuples defining the wall.
        frequency (float): The frequency of the signal in Hz.

    Returns:
        complex: The complex UTD diffraction coefficient.
    """
    # --- Placeholder for implementation ---
    # 1. Calculate wavenumber k
    wavelength = SPEED_OF_LIGHT / frequency
    k = 2 * np.pi / wavelength
    print(f"Wavelength (λ): {wavelength:.4f} m, Wavenumber (k): {k:.4f}")

    # 2. Establish local coordinate system at the edge
    # The reference plane is the face of the wall.
    # We need to find the other point of the wall that is not the edge.
    p1, p2 = wall_points
    face_vector = np.array(p2) - np.array(p1) if np.allclose(edge, p1) else np.array(p1) - np.array(p2)
    face_angle = np.arctan2(face_vector[1], face_vector[0])
    print(f"Wall face angle: {np.degrees(face_angle):.2f}°")

    # 3. Calculate angles phi and phi_0 relative to the wall face
    # Vector from edge to source (for incident ray)
    incident_vector = np.array(source) - np.array(edge)
    phi_0 = np.arctan2(incident_vector[1], incident_vector[0]) - face_angle
    # Normalize to [0, 2*pi]
    phi_0 = phi_0 % (2 * np.pi)
    print(f"Incident angle (φ₀): {np.degrees(phi_0):.2f}°")
    
    # Vector from edge to observer (for diffracted ray)
    diffracted_vector = np.array(observer) - np.array(edge)
    phi = np.arctan2(diffracted_vector[1], diffracted_vector[0]) - face_angle
    # Normalize to [0, 2*pi]
    phi = phi % (2 * np.pi)
    print(f"Diffraction angle (φ): {np.degrees(phi):.2f}°")


    # ... The rest of the UTD steps will go here ...
    
    print("UTD calculation partially implemented.")
    # Return a default complex value for D, and the calculated angles in radians
    return 1.0 + 0.0j, phi_0, phi

if __name__ == '__main__':
    # --- Example usage for testing ---
    # We will add test cases here as we build the function.
    print("--- Running Test Case ---")
    
    # Define a simple scenario: Horizontal wall
    source_pos = (10, 20)
    wall = ((30, 10), (70, 10))
    diffracting_edge = wall[0]
    observer_pos = (50, -10) # In the shadow region
    freq = 900e6 # 900 MHz
    
    # The function now returns 3 values, so we capture them for testing.
    coeff, phi_0_rad, phi_rad = calculate_utd_coefficient(source_pos, diffracting_edge, observer_pos, wall, freq)
    print(f"Returned values: D={coeff}, φ₀={np.degrees(phi_0_rad):.2f}°, φ={np.degrees(phi_rad):.2f}°")
    
    print("\n--- Test Case Complete ---")
    print("utd_physics.py - Ready for next steps.") 