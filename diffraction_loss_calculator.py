import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import time

class DiffractionLossCalculator:
    def __init__(self):
        """Initialize the diffraction loss calculator with default parameters."""
        # Simulation parameters
        self.sim_params = {
            'tx': {'x': 50, 'y': 30},
            'rx': {'x': 450, 'y': 30},
            'wall': {'x': 250, 'height': 35},
            'world': {'width': 500, 'height': 120},
            'freq_MHz': 900,
            'ground_height': 20  # Ground level
        }
        
        # Physical constants
        self.C = 299792458  # Speed of light in m/s
        
        # Visualization settings
        self.setup_visualization()
        
        # Heatmap cache for performance
        self.heatmap_cache = {}
        
    def setup_visualization(self):
        """Set up the interactive matplotlib visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 12))
        
        # Main visualization (top)
        self.ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Heatmap (bottom left)
        self.ax_heatmap = plt.subplot2grid((3, 2), (2, 0))
        
        # Info panel (bottom right)
        self.ax_info = plt.subplot2grid((3, 2), (2, 1))
        self.ax_info.axis('off')
        
        # Add sliders
        self.setup_sliders()
        
        # Initialize plot
        self.update_visualization()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.dragging = False
        self.drag_target = None
        
    def setup_sliders(self):
        """Create interactive sliders for parameters."""
        # Adjust subplot to make room for sliders
        plt.subplots_adjust(bottom=0.35)
        
        # Slider positions
        slider_height = 0.03
        slider_spacing = 0.04
        slider_left = 0.1
        slider_width = 0.35
        
        # TX position sliders
        ax_tx_x = plt.axes([slider_left, 0.25, slider_width, slider_height])
        ax_tx_y = plt.axes([slider_left, 0.21, slider_width, slider_height])
        
        self.slider_tx_x = Slider(ax_tx_x, 'TX X', 10, 490, 
                                 valinit=self.sim_params['tx']['x'], valstep=1)
        self.slider_tx_y = Slider(ax_tx_y, 'TX Y', 5, 115, 
                                 valinit=self.sim_params['tx']['y'], valstep=1)
        
        # RX position sliders
        ax_rx_x = plt.axes([slider_left, 0.16, slider_width, slider_height])
        ax_rx_y = plt.axes([slider_left, 0.12, slider_width, slider_height])
        
        self.slider_rx_x = Slider(ax_rx_x, 'RX X', 10, 490, 
                                 valinit=self.sim_params['rx']['x'], valstep=1)
        self.slider_rx_y = Slider(ax_rx_y, 'RX Y', 5, 115, 
                                 valinit=self.sim_params['rx']['y'], valstep=1)
        
        # Wall parameters
        ax_wall_x = plt.axes([slider_left + 0.45, 0.25, slider_width, slider_height])
        ax_wall_h = plt.axes([slider_left + 0.45, 0.21, slider_width, slider_height])
        
        self.slider_wall_x = Slider(ax_wall_x, 'Wall X', 50, 450, 
                                   valinit=self.sim_params['wall']['x'], valstep=1)
        self.slider_wall_h = Slider(ax_wall_h, 'Wall H', 5, 100, 
                                   valinit=self.sim_params['wall']['height'], valstep=1)
        
        # Frequency slider
        ax_freq = plt.axes([slider_left + 0.45, 0.16, slider_width, slider_height])
        self.slider_freq = Slider(ax_freq, 'Freq (MHz)', 100, 6000, 
                                 valinit=self.sim_params['freq_MHz'], valstep=10)
        
        # Connect slider events
        self.slider_tx_x.on_changed(self.update_params)
        self.slider_tx_y.on_changed(self.update_params)
        self.slider_rx_x.on_changed(self.update_params)
        self.slider_rx_y.on_changed(self.update_params)
        self.slider_wall_x.on_changed(self.update_params)
        self.slider_wall_h.on_changed(self.update_params)
        self.slider_freq.on_changed(self.update_params)
        
        # Reset button
        ax_reset = plt.axes([slider_left + 0.45, 0.08, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_params)
        
    def calculate_diffraction(self, rx_pos, freq_MHz, wall_height):
        """
        Calculate diffraction loss for given receiver position.
        
        Args:
            rx_pos: Dictionary with 'x' and 'y' keys
            freq_MHz: Frequency in MHz
            wall_height: Wall height in meters
            
        Returns:
            Dictionary with diffraction calculation results
        """
        # Calculate wavelength
        lambda_m = self.C / (freq_MHz * 1e6)
        
        # Calculate distances
        d1 = self.sim_params['wall']['x'] - self.sim_params['tx']['x']
        d2 = rx_pos['x'] - self.sim_params['wall']['x']
        
        # Check for invalid geometry
        if d1 <= 0 or d2 <= 0:
            return {
                'h': 0,
                'v': -np.inf,
                'diffraction_loss_dB': 0,
                'fspl_dB': 0,
                'total_loss_dB': 0,
                'f1_radius': 0
            }
        
        # Calculate line-of-sight height at wall
        tx_y = self.sim_params['tx']['y']
        rx_y = rx_pos['y']
        los_height_at_wall = tx_y + (rx_y - tx_y) * (d1 / (d1 + d2))
        
        # Calculate diffraction parameter
        h = wall_height - los_height_at_wall
        v = h * np.sqrt((2 / lambda_m) * ((d1 + d2) / (d1 * d2)))
        
        # Calculate diffraction loss based on Fresnel-Kirchhoff theory
        if v <= -1:
            diffraction_loss_dB = 0
        elif v <= 0:
            diffraction_loss_dB = 20 * np.log10(0.5 - 0.62 * v)
        elif v <= 1:
            diffraction_loss_dB = 20 * np.log10(0.5 * np.exp(-0.95 * v))
        elif v <= 2.4:
            diffraction_loss_dB = 20 * np.log10(0.4 - np.sqrt(0.1184 - (0.38 - 0.1 * v)**2))
        else:
            diffraction_loss_dB = 20 * np.log10(0.225 / v)
        
        # Make it a loss (negative becomes positive)
        diffraction_loss_dB = -diffraction_loss_dB
        
        # Calculate free space path loss
        total_dist = np.sqrt((rx_pos['x'] - self.sim_params['tx']['x'])**2 + 
                            (rx_pos['y'] - self.sim_params['tx']['y'])**2)
        dist_clamped = max(1.0, total_dist)
        fspl_dB = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55
        
        # Total loss
        total_loss_dB = fspl_dB + diffraction_loss_dB
        
        # First Fresnel zone radius
        f1_radius = np.sqrt((lambda_m * d1 * d2) / (d1 + d2))
        
        return {
            'h': h,
            'v': v,
            'diffraction_loss_dB': diffraction_loss_dB,
            'fspl_dB': fspl_dB,
            'total_loss_dB': total_loss_dB,
            'f1_radius': f1_radius
        }
    
    def generate_heatmap(self, resolution=50):
        """Generate 2D heatmap of diffraction loss."""
        # Create cache key
        cache_key = (
            self.sim_params['tx']['x'], self.sim_params['tx']['y'],
            self.sim_params['wall']['x'], self.sim_params['wall']['height'],
            self.sim_params['freq_MHz'], resolution
        )
        
        # Check cache
        if cache_key in self.heatmap_cache:
            return self.heatmap_cache[cache_key]
        
        # Generate grid
        x_grid = np.linspace(10, 490, resolution)
        y_grid = np.linspace(5, 115, resolution)
        
        # Calculate diffraction loss for each point
        loss_grid = np.zeros((len(y_grid), len(x_grid)))
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                rx_pos = {'x': x, 'y': y}
                result = self.calculate_diffraction(rx_pos, self.sim_params['freq_MHz'], 
                                                  self.sim_params['wall']['height'])
                loss_grid[i, j] = result['total_loss_dB']
        
        # Cache result
        self.heatmap_cache[cache_key] = (x_grid, y_grid, loss_grid)
        
        return x_grid, y_grid, loss_grid
    
    def draw_main_visualization(self):
        """Draw the main diffraction visualization."""
        self.ax_main.clear()
        
        # Get current parameters
        tx = self.sim_params['tx']
        rx = self.sim_params['rx']
        wall = self.sim_params['wall']
        world = self.sim_params['world']
        freq_MHz = self.sim_params['freq_MHz']
        
        # Calculate diffraction for current receiver position
        results = self.calculate_diffraction(rx, freq_MHz, wall['height'])
        
        # Set up coordinate system
        self.ax_main.set_xlim(0, world['width'])
        self.ax_main.set_ylim(0, world['height'])
        self.ax_main.set_aspect('equal')
        
        # Draw ground
        ground_y = self.sim_params['ground_height']
        self.ax_main.fill_between([0, world['width']], [0, 0], [ground_y, ground_y], 
                                 color='#e2e8f0', alpha=0.5, label='Ground')
        
        # Draw sky
        self.ax_main.fill_between([0, world['width']], [ground_y, ground_y], 
                                 [world['height'], world['height']], 
                                 color='#f8fafc', alpha=0.3)
        
        # Draw Fresnel zone if valid
        if results['v'] != -np.inf and results['f1_radius'] > 0:
            d1 = wall['x'] - tx['x']
            d2 = rx['x'] - wall['x']
            
            center_x = (tx['x'] + rx['x']) / 2
            center_y = (tx['y'] + rx['y']) / 2
            radius_a = (d1 + d2) / 2
            radius_b = results['f1_radius']
            
            angle = np.arctan2(rx['y'] - tx['y'], rx['x'] - tx['x'])
            
            # Create Fresnel zone ellipse
            ellipse = Ellipse((center_x, center_y), 2*radius_a, 2*radius_b, 
                             angle=np.degrees(angle), 
                             facecolor='lightgreen', alpha=0.2, 
                             edgecolor='green', linewidth=2)
            self.ax_main.add_patch(ellipse)
            
            # Label Fresnel zone
            self.ax_main.text(wall['x'], wall['height'] + 5, 
                             f'F₁ Radius: {results["f1_radius"]:.2f}m',
                             ha='center', va='bottom', fontsize=12)
        
        # Draw wall
        wall_color = '#475569'
        self.ax_main.add_patch(patches.Rectangle(
            (wall['x'] - 2, ground_y), 4, wall['height'] - ground_y,
            facecolor=wall_color, edgecolor='black', linewidth=2
        ))
        
        # Draw line of sight
        los_color = '#ef4444' if results['h'] > 0 else '#22c55e'
        self.ax_main.plot([tx['x'], rx['x']], [tx['y'], rx['y']], 
                         color=los_color, linewidth=2, linestyle='--', alpha=0.7,
                         label='Line of Sight')
        
        # Draw diffraction path
        self.ax_main.plot([tx['x'], wall['x'], rx['x']], 
                         [tx['y'], wall['height'], rx['y']], 
                         color='#0ea5e9', linewidth=3, alpha=0.8,
                         label='Diffraction Path')
        
        # Draw transmitter and receiver
        self.ax_main.plot(tx['x'], tx['y'], 'o', color='#1d4ed8', 
                         markersize=15, label='Transmitter')
        self.ax_main.plot(rx['x'], rx['y'], 'o', color='#be123c', 
                         markersize=15, label='Receiver')
        
        # Add labels
        self.ax_main.text(tx['x'], tx['y'] + 8, 'TX', ha='center', va='bottom', 
                         fontsize=12, fontweight='bold')
        self.ax_main.text(rx['x'], rx['y'] + 8, 'RX', ha='center', va='bottom', 
                         fontsize=12, fontweight='bold')
        
        # Set labels and title
        self.ax_main.set_xlabel('Distance (m)', fontsize=14)
        self.ax_main.set_ylabel('Height (m)', fontsize=14)
        self.ax_main.set_title('Diffraction Loss Visualization', fontsize=16, fontweight='bold')
        self.ax_main.legend(loc='upper right')
        self.ax_main.grid(True, alpha=0.3)
        
        return results
    
    def draw_heatmap(self):
        """Draw the 2D heatmap of diffraction loss."""
        self.ax_heatmap.clear()
        
        # Generate heatmap data
        x_grid, y_grid, loss_grid = self.generate_heatmap(resolution=40)
        
        # Create custom colormap
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('diffraction', colors, N=n_bins)
        
        # Plot heatmap
        im = self.ax_heatmap.imshow(loss_grid, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                                   origin='lower', cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=self.ax_heatmap, shrink=0.6)
        cbar.set_label('Total Loss (dB)', fontsize=12)
        
        # Mark current positions
        tx = self.sim_params['tx']
        rx = self.sim_params['rx']
        wall = self.sim_params['wall']
        
        self.ax_heatmap.plot(tx['x'], tx['y'], 'wo', markersize=8, 
                           markeredgecolor='black', markeredgewidth=2, label='TX')
        self.ax_heatmap.plot(rx['x'], rx['y'], 'wo', markersize=8, 
                           markeredgecolor='black', markeredgewidth=2, label='RX')
        self.ax_heatmap.axvline(x=wall['x'], color='white', linewidth=3, 
                              alpha=0.8, label='Wall')
        
        # Labels
        self.ax_heatmap.set_xlabel('Distance (m)', fontsize=12)
        self.ax_heatmap.set_ylabel('Height (m)', fontsize=12)
        self.ax_heatmap.set_title('Diffraction Loss Heatmap', fontsize=14)
        self.ax_heatmap.legend(loc='upper right')
        
    def update_info_panel(self, results):
        """Update the information panel with current results."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Create info text
        info_text = f"""
Diffraction Analysis:
─────────────────────

Geometry:
• TX Position: ({self.sim_params['tx']['x']:.0f}, {self.sim_params['tx']['y']:.0f}) m
• RX Position: ({self.sim_params['rx']['x']:.0f}, {self.sim_params['rx']['y']:.0f}) m
• Wall: x={self.sim_params['wall']['x']:.0f}m, h={self.sim_params['wall']['height']:.0f}m
• Frequency: {self.sim_params['freq_MHz']:.0f} MHz

Diffraction Parameters:
• Clearance (h): {results['h']:.2f} m
• Fresnel Parameter (v): {results['v']:.2f}
• F₁ Radius: {results['f1_radius']:.2f} m

Path Loss Components:
• Free Space Loss: {results['fspl_dB']:.2f} dB
• Diffraction Loss: {results['diffraction_loss_dB']:.2f} dB
• Total Loss: {results['total_loss_dB']:.2f} dB

Signal Quality:
"""
        
        # Add signal quality assessment
        if results['diffraction_loss_dB'] < 3:
            info_text += "• Status: GOOD (minimal diffraction)"
        elif results['diffraction_loss_dB'] < 10:
            info_text += "• Status: FAIR (moderate diffraction)"
        else:
            info_text += "• Status: POOR (significant diffraction)"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes, 
                         fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    def update_visualization(self):
        """Update the complete visualization."""
        # Draw main visualization and get results
        results = self.draw_main_visualization()
        
        # Draw heatmap
        self.draw_heatmap()
        
        # Update info panel
        self.update_info_panel(results)
        
        # Refresh display
        self.fig.canvas.draw_idle()
    
    def update_params(self, val):
        """Update parameters from sliders."""
        self.sim_params['tx']['x'] = self.slider_tx_x.val
        self.sim_params['tx']['y'] = self.slider_tx_y.val
        self.sim_params['rx']['x'] = self.slider_rx_x.val
        self.sim_params['rx']['y'] = self.slider_rx_y.val
        self.sim_params['wall']['x'] = self.slider_wall_x.val
        self.sim_params['wall']['height'] = self.slider_wall_h.val
        self.sim_params['freq_MHz'] = self.slider_freq.val
        
        # Clear heatmap cache when parameters change
        self.heatmap_cache.clear()
        
        # Update visualization
        self.update_visualization()
    
    def reset_params(self, event):
        """Reset parameters to defaults."""
        self.sim_params = {
            'tx': {'x': 50, 'y': 30},
            'rx': {'x': 450, 'y': 30},
            'wall': {'x': 250, 'height': 35},
            'world': {'width': 500, 'height': 120},
            'freq_MHz': 900,
            'ground_height': 20
        }
        
        # Update sliders
        self.slider_tx_x.reset()
        self.slider_tx_y.reset()
        self.slider_rx_x.reset()
        self.slider_rx_y.reset()
        self.slider_wall_x.reset()
        self.slider_wall_h.reset()
        self.slider_freq.reset()
        
        # Clear cache and update
        self.heatmap_cache.clear()
        self.update_visualization()
    
    def on_click(self, event):
        """Handle mouse click events for dragging."""
        if event.inaxes != self.ax_main:
            return
            
        # Check if click is near transmitter or receiver
        tx_dist = np.sqrt((event.xdata - self.sim_params['tx']['x'])**2 + 
                         (event.ydata - self.sim_params['tx']['y'])**2)
        rx_dist = np.sqrt((event.xdata - self.sim_params['rx']['x'])**2 + 
                         (event.ydata - self.sim_params['rx']['y'])**2)
        
        if tx_dist < 20:
            self.dragging = True
            self.drag_target = 'tx'
        elif rx_dist < 20:
            self.dragging = True
            self.drag_target = 'rx'
    
    def on_drag(self, event):
        """Handle mouse drag events."""
        if not self.dragging or event.inaxes != self.ax_main:
            return
            
        # Update position
        if self.drag_target == 'tx':
            self.sim_params['tx']['x'] = max(10, min(490, event.xdata))
            self.sim_params['tx']['y'] = max(5, min(115, event.ydata))
            self.slider_tx_x.set_val(self.sim_params['tx']['x'])
            self.slider_tx_y.set_val(self.sim_params['tx']['y'])
        elif self.drag_target == 'rx':
            self.sim_params['rx']['x'] = max(10, min(490, event.xdata))
            self.sim_params['rx']['y'] = max(5, min(115, event.ydata))
            self.slider_rx_x.set_val(self.sim_params['rx']['x'])
            self.slider_rx_y.set_val(self.sim_params['rx']['y'])
        
        # Clear cache and update
        self.heatmap_cache.clear()
        self.update_visualization()
    
    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging = False
        self.drag_target = None
    
    def show(self):
        """Show the interactive visualization."""
        plt.show()


def main():
    """Main function to run the diffraction loss calculator."""
    print("🔊 Diffraction Loss Calculator")
    print("=" * 50)
    print("Interactive controls:")
    print("• Use sliders to adjust parameters")
    print("• Click and drag TX/RX to move them")
    print("• Watch heatmap update in real-time")
    print("• Reset button restores defaults")
    print("=" * 50)
    
    # Create and run calculator
    calculator = DiffractionLossCalculator()
    calculator.show()


if __name__ == "__main__":
    main() 