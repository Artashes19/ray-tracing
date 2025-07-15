import matplotlib
matplotlib.use('WebAgg') # Use the WebAgg backend for browser-based visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

class GTDVisualizer:
    """
    An interactive visualizer for the Geometrical Theory of Diffraction (GTD).

    This class sets up a matplotlib plot to visualize walls, a transmitter,
    and the ray tracing process, including incident and diffracted rays.
    """
    def __init__(self, grid_size=(100, 100)):
        """
        Initializes the visualizer environment.
        
        Args:
            grid_size (tuple): The (width, height) of the simulation grid.
        """
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_title("GTD Visualizer - Left-click to draw walls, Right-click for transmitter")
        self.ax.grid(True)
        
        # Add a Reset button
        reset_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04]) # x, y, width, height
        self.reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_scene)

        self.wall_points = []
        self.edges = []
        self.transmitter = None
        self.mouse_pos = None
        self.target_points = []
        self.ray_artist = None
        self.snap_threshold = 5
        self.diffraction_events = [] # To store (edge, target) tuples
        
        # This will store tuples of (type, data)
        # e.g., ('clear', (x,y)), ('blocked', (x,y)), ('diffraction', (edge_point, target_point))
        self.fired_rays = []
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print("Visualizer initialized. Ready for user input.")

    def reset_scene(self, *args):
        """Resets the entire scene to its initial state."""
        self.wall_points = []
        self.edges = []
        self.transmitter = None
        self.target_points = []
        self.diffraction_events = []
        self.fired_rays = []
        
        self.draw_scene()
        print("Scene has been reset.")

    def recalculate_rays(self, single_target=None):
        """
        Recalculates ray statuses. If single_target is provided, it only
        calculates for that new target and appends it. Otherwise, it
        recalculates all existing rays.
        """
        if not self.transmitter:
            return

        targets_to_process = [single_target] if single_target else self.target_points
        if not single_target:
            self.fired_rays = [] # Clear all rays if we are doing a full recalc

        if len(self.wall_points) < 2:
            # If there's no wall, all new rays are clear
            for target in targets_to_process:
                if target: self.fired_rays.append(('clear', target))
            return

        for target in targets_to_process:
            if not target: continue
            is_blocked = self.check_intersection(self.transmitter, target, self.wall_points[0], self.wall_points[1])
            if is_blocked:
                self.fired_rays.append(('blocked', target))
            else:
                hit_edge = None
                if self.edges:
                    for edge in self.edges:
                        if self.distance_point_to_segment(edge, self.transmitter, target) < 2.0:
                            hit_edge = edge
                            break
                if hit_edge:
                    self.fired_rays.append(('diffraction', (hit_edge, target)))
                else:
                    self.fired_rays.append(('clear', target))

    def on_key_press(self, event):
        """Handles key press events."""
        # Spacebar to add rays
        if event.key == ' ':
            if self.transmitter and self.mouse_pos:
                target = self.mouse_pos
                self.target_points.append(target) # Add to the permanent list of targets
                
                # Immediately calculate the status of the new ray and add it
                self.recalculate_rays(single_target=target)
                self.draw_scene()
        
        # 'r' key to reset the scene (keeping it as a backup)
        elif event.key == 'r':
            self.reset_scene()

    def on_mouse_move(self, event):
        """Handles mouse movement to draw a ray from the transmitter."""
        if not event.inaxes or not self.transmitter:
            return
        self.mouse_pos = (event.xdata, event.ydata)
        self.draw_scene()

    def on_click(self, event):
        """Handles mouse click events to place objects in the scene."""
        if event.inaxes != self.ax:
            return
        
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        if event.button == 1:  # Left-click for wall
            if len(self.wall_points) >= 2:
                self.wall_points = []  # Start a new wall
                self.edges = []
                # Don't clear fired_rays here, recalculate upon completion
            self.wall_points.append((ix, iy))
            print(f"Added wall point: ({ix}, {iy}). Total points: {len(self.wall_points)}")

            if len(self.wall_points) == 2:
                p1_raw, p2_raw = self.wall_points
                dx = abs(p1_raw[0] - p2_raw[0])
                dy = abs(p1_raw[1] - p2_raw[1])

                if dx < self.snap_threshold:
                    # Snap to vertical
                    self.wall_points = [p1_raw, (p1_raw[0], p2_raw[1])]
                    print("Snapped to vertical wall.")
                elif dy < self.snap_threshold:
                    # Snap to horizontal
                    self.wall_points = [p1_raw, (p2_raw[0], p1_raw[1])]
                    print("Snapped to horizontal wall.")
                else:
                    print("Created inclined wall.")

                self.detect_edges()
                self.recalculate_rays() # Recalculate all rays against the new wall
            else:
                # If we are starting a new wall, just redraw
                self.draw_scene()

        elif event.button == 3:  # Right-click for transmitter
            self.transmitter = (ix, iy)
            print(f"Transmitter placed at: {self.transmitter}")

        self.draw_scene()

    def detect_edges(self):
        """Identifies the corner points (edges) of the current wall."""
        if len(self.wall_points) == 2:
            self.edges = self.wall_points[:] # Make a copy
            print(f"Detected edges at: {self.edges}")
        else:
            self.edges = []

    def distance_point_to_segment(self, p, a, b):
        """Calculates the shortest distance from point p to line segment (a, b)."""
        px, py = p
        ax, ay = a
        bx, by = b

        # Vector from a to b
        dx = bx - ax
        dy = by - ay

        if dx == 0 and dy == 0:  # Segment is a point
            return np.sqrt((px - ax)**2 + (py - ay)**2)

        # Projection of p onto the line ab
        t = ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2)
        
        # Clamp t to the range [0, 1] to stay on the segment
        t = max(0, min(1, t))

        # Coordinates of the closest point on the segment
        closest_x = ax + t * dx
        closest_y = ay + t * dy

        # Return the distance
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def _orientation(self, p, q, r):
        """Helper for check_intersection to find orientation of ordered triplet (p, q, r)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
              (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def check_intersection(self, p1, p2, p3, p4):
        """
        Checks if line segment (p1, p2) intersects with (p3, p4).
        This is a standard line-segment intersection algorithm.
        """
        o1 = self._orientation(p1, p2, p3)
        o2 = self._orientation(p1, p2, p4)
        o3 = self._orientation(p3, p4, p1)
        o4 = self._orientation(p3, p4, p2)

        if o1 != o2 and o3 != o4:
            return True
            
        # We don't need to handle the special collinear case for this visualizer.
        return False

    def draw_scene(self):
        """Clears the plot and redraws all scene elements."""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_title("GTD Visualizer - Left-click to draw walls, Right-click for transmitter")
        self.ax.grid(True)

        # Draw transmitter
        if self.transmitter:
            self.ax.plot(self.transmitter[0], self.transmitter[1], 'ro', markersize=8, label='Transmitter')

        # Draw wall
        if len(self.wall_points) == 1:
            p1 = self.wall_points[0]
            self.ax.plot(p1[0], p1[1], 'ko', markersize=4, label='Wall Start')
        elif len(self.wall_points) == 2:
            p1, p2 = self.wall_points
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3, label='Wall')

        # Draw detected edges
        if self.edges:
            edge_x, edge_y = zip(*self.edges)
            self.ax.plot(edge_x, edge_y, 'bo', markersize=10, label='Edges', alpha=0.5)

        # Draw static (fired) rays
        if self.transmitter: # Ensure transmitter exists before drawing rays
            # Handle fired rays
            for ray_type, data in self.fired_rays:
                if ray_type == 'clear':
                    target = data
                    self.ax.plot([self.transmitter[0], target[0]], [self.transmitter[1], target[1]], color='g', linestyle='-', label='Fired Ray')
                elif ray_type == 'blocked':
                    target = data
                    self.ax.plot([self.transmitter[0], target[0]], [self.transmitter[1], target[1]], color='r', linestyle='-', label='Fired Ray')
                elif ray_type == 'diffraction':
                    edge, target = data
                    # Draw the real part of the ray
                    tx, ty = self.transmitter
                    dx, dy = target[0] - tx, target[1] - ty
                    mag = np.sqrt(dx**2 + dy**2)
                    if mag > 0:
                        dx /= mag
                        dy /= mag
                    
                    ex, ey = edge
                    proj_dist = (ex - tx) * dx + (ey - ty) * dy
                    diffraction_point = (tx + proj_dist * dx, ty + proj_dist * dy)
                    
                    self.ax.plot([tx, diffraction_point[0]], [ty, diffraction_point[1]], color='y', linestyle='-', label='Fired Ray')
                    # Draw the shadowed part
                    self.ax.plot([diffraction_point[0], target[0]], [diffraction_point[1], target[1]],
                                 color='gray', linestyle=':', label='Shadowed Path')
        
        self.draw_diffracted_rays()

        # Draw dynamic ray from transmitter to mouse
        if self.transmitter and self.mouse_pos:
            ray_color = 'g'
            if len(self.wall_points) == 2:
                # We can also color the dynamic ray for edge proximity, but we'll keep it simple for now
                is_intersecting = self.check_intersection(self.transmitter, self.mouse_pos, self.wall_points[0], self.wall_points[1])
                
                # Check for edge proximity on the dynamic ray
                is_near_edge = False
                if not is_intersecting and self.edges:
                    for edge in self.edges:
                        if self.distance_point_to_segment(edge, self.transmitter, self.mouse_pos) < 2.0:
                            is_near_edge = True
                            break
                
                if is_intersecting:
                    ray_color = 'r'
                elif is_near_edge:
                    ray_color = 'y'
            
            p1, p2 = self.transmitter, self.mouse_pos
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ray_color, linestyle='--', label='Incident Ray')

        if self.transmitter or self.wall_points:
             # Place legend outside the plot area, to the upper right
             self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
       
        self.fig.canvas.draw_idle()

    def draw_diffracted_rays(self):
        """Draws a fan of rays from each diffraction event."""
        # Guard clause: only run if a full wall exists
        if len(self.wall_points) < 2 or not self.transmitter:
            return

        wall_p1, wall_p2 = self.wall_points

        diffraction_events = [data for type, data in self.fired_rays if type == 'diffraction']

        for edge, _ in diffraction_events:
            num_rays = 12
            length = 40
            for i in range(num_rays):
                angle = 2 * np.pi * i / num_rays
                end_x = edge[0] + length * np.cos(angle)
                end_y = edge[1] + length * np.sin(angle)
                
                # Check if the diffracted ray would immediately go into the wall
                # Move start point slightly away from the edge to avoid self-intersection
                epsilon = 0.1
                start_x = edge[0] + epsilon * np.cos(angle)
                start_y = edge[1] + epsilon * np.sin(angle)
                
                if not self.check_intersection((start_x, start_y), (end_x, end_y), wall_p1, wall_p2):
                    self.ax.plot([edge[0], end_x], [edge[1], end_y], color='m', linestyle='-', label='Diffracted Ray')

    def run(self):
        """Displays the matplotlib window."""
        self.fig.subplots_adjust(right=0.75) # Make space for the legend
        plt.show()

if __name__ == '__main__':
    print("Starting the GTD Visualizer...")
    print("When the script runs, open the URL it provides in your browser.")
    vis = GTDVisualizer()
    vis.run() 