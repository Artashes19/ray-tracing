import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
import utd_physics

def visualize_angles(source, wall, edge_index, observer, freq):
    """
    Creates a static plot to visualize the geometry and angles of a diffraction event.

    Args:
        source (tuple): (x, y) of the transmitter.
        wall (tuple): A tuple of two (x, y) points defining the wall.
        edge_index (int): 0 or 1, indicating which wall point is the edge.
        observer (tuple): (x, y) of the observation point.
        freq (float): The signal frequency in Hz.
    """
    edge = wall[edge_index]
    
    # --- Physics Calculation ---
    # We capture all four return values.
    _, vis_phi_0_rad, phi_rad, face_angle_ref_rad = utd_physics.calculate_utd_coefficient(
        source, edge, observer, wall, freq
    )
    # For visualization, we use the angles as they are now returned:
    # - vis_phi_0_rad is the smallest angle for the incident ray
    # - phi_rad is the CCW angle from the 0-face for the diffracted ray
    vis_phi_0_deg = np.degrees(vis_phi_0_rad)
    phi_deg = np.degrees(phi_rad)
    face_angle_deg = np.degrees(face_angle_ref_rad)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f'Diffraction Angle Visualization (Freq: {freq/1e6} MHz)')
    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Y-coordinate (m)")

    # Plot scene elements
    ax.plot(*zip(*wall), 'k-', linewidth=3, label='Wall')
    ax.plot(source[0], source[1], 'ro', markersize=10, label='Source')
    ax.plot(observer[0], observer[1], 'go', markersize=10, label='Observer')
    ax.plot(edge[0], edge[1], 'bo', markersize=12, label='Edge', alpha=0.7)

    # Plot rays
    ax.plot([source[0], edge[0]], [source[1], edge[1]], 'r--', label='Incident Ray')
    ax.plot([edge[0], observer[0]], [edge[1], observer[1]], 'g--', label='Diffracted Ray')
    
    # --- Angle Visualization ---
    # We no longer calculate this here; we use the one from the physics module.
    # p1_wall, p2_wall = wall
    # face_vector = np.array(p2_wall) - np.array(p1_wall) if edge_index == 0 else np.array(p1_wall) - np.array(p2_wall)
    # face_angle_deg = np.degrees(np.arctan2(face_vector[1], face_vector[0]))

    # Reference line for angles (the 0-face)
    ref_line_end = (edge[0] + 50 * np.cos(np.radians(face_angle_deg)), edge[1] + 50 * np.sin(np.radians(face_angle_deg)))
    ax.plot([edge[0], ref_line_end[0]], [edge[1], ref_line_end[1]], 'k:', alpha=0.5, label='0-Face Reference')
    
    # Draw incident angle arc - using the smallest angle
    arc_inc = Arc(edge, 100, 100, angle=face_angle_deg, theta1=0, theta2=vis_phi_0_deg,
                  color='blue', linewidth=2, linestyle='--', label=f'Incident Angle ($\\phi_0$ = {vis_phi_0_deg:.1f}°)')
    ax.add_patch(arc_inc)
    
    # Draw diffraction angle arc - using the formula angle
    arc_diff = Arc(edge, 80, 80, angle=face_angle_deg, theta1=0, theta2=phi_deg,
                   color='purple', linewidth=2, linestyle='--', label=f'Diffraction Angle ($\\phi$ = {phi_deg:.1f}°)')
    ax.add_patch(arc_diff)

    # Auto-adjust limits
    all_x = [p[0] for p in [source, observer, *wall]]
    all_y = [p[1] for p in [source, observer, *wall]]
    ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
    ax.set_ylim(min(all_y) - 20, max(all_y) + 20)
    
    ax.legend()
    # plt.show() does not work well in remote environments without a display.
    # Instead, we'll save the plot to a file.
    plt.savefig("diffraction_plot.png")
    plt.close(fig) # Free up memory

if __name__ == '__main__':
    # --- Define a Test Scenario ---
    # TEST CASE 3
    source_pos = (100, 100)
    wall_geom = ((100, 120), (50, 120))
    diffracting_edge_idx = 0
    observer_pos = (100, 180)
    frequency = 900e6  # 900 MHz
    
    print("--- Running Test Case 3: Complex Geometry ---")
    visualize_angles(source_pos, wall_geom, diffracting_edge_idx, observer_pos, frequency)
    print("--- Plot saved to diffraction_plot.png ---") 