import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from diffraction_loss_simple import DiffractionLossCalculator
import io
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

class DiffractionHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.calc = DiffractionLossCalculator()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Diffraction Loss Calculator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .control-group { flex: 1; min-width: 200px; }
        .slider-container { margin: 10px 0; }
        .slider { width: 100%; }
        .value-display { font-weight: bold; color: #333; }
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-image { max-width: 100%; border: 1px solid #ccc; border-radius: 5px; }
        .instructions { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .progress-bar { width: 100%; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5722); transition: width 0.3s; }
        .coords { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .update-btn { background: #4CAF50; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .update-btn:hover { background: #45a049; }
        .reset-btn { background: #ff6b6b; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔊 Interactive Diffraction Loss Calculator</h1>
        
        <div class="instructions">
            <h3>Instructions:</h3>
            <ul>
                <li>Adjust sliders to change parameters</li>
                <li>Enter TX/RX coordinates manually</li>
                <li>Click "Update Plot" to see changes</li>
                <li>Wall height progress bar shows current percentage</li>
            </ul>
        </div>

        <div class="controls">
            <div class="control-group">
                <h3>Wall Parameters</h3>
                <div class="slider-container">
                    <label>Wall Height: <span class="value-display" id="wall-height-value">35 m</span></label>
                    <input type="range" id="wall-height" class="slider" min="10" max="80" value="35" step="1">
                    <div class="progress-bar">
                        <div class="progress-fill" id="wall-progress" style="width: 43.75%;"></div>
                    </div>
                </div>
                <div class="slider-container">
                    <label>Wall X-Position: <span class="value-display" id="wall-x-value">250 m</span></label>
                    <input type="range" id="wall-x" class="slider" min="100" max="400" value="250" step="5">
                </div>
            </div>
            
            <div class="control-group">
                <h3>Signal Parameters</h3>
                <div class="slider-container">
                    <label>Frequency: <span class="value-display" id="freq-value">900 MHz</span></label>
                    <input type="range" id="frequency" class="slider" min="100" max="6000" value="900" step="50">
                </div>
            </div>
            
            <div class="control-group">
                <h3>Positions</h3>
                <div class="coords">
                    <label>TX Position:</label><br>
                    X: <input type="number" id="tx-x" value="100" step="1" style="width: 60px;">
                    Y: <input type="number" id="tx-y" value="40" step="1" style="width: 60px;">
                </div>
                <div class="coords">
                    <label>RX Position:</label><br>
                    X: <input type="number" id="rx-x" value="400" step="1" style="width: 60px;">
                    Y: <input type="number" id="rx-y" value="40" step="1" style="width: 60px;">
                </div>
                <button class="update-btn" onclick="updatePlot()">Update Plot</button>
                <button class="reset-btn" onclick="resetValues()">Reset</button>
            </div>
        </div>

        <div class="plot-container">
            <img id="plot-image" class="plot-image" src="/plot?tx_x=100&tx_y=40&rx_x=400&rx_y=40&wall_x=250&wall_height=35&freq_MHz=900" alt="Diffraction Plot">
        </div>

        <div id="results" style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h3>Live Results:</h3>
            <p id="results-text">Click "Update Plot" to see calculations...</p>
        </div>
    </div>

    <script>
        // Update slider values
        document.getElementById('wall-height').addEventListener('input', function(e) {
            const value = e.target.value;
            document.getElementById('wall-height-value').textContent = value + ' m';
            const percentage = (value / 80) * 100;
            document.getElementById('wall-progress').style.width = percentage + '%';
        });

        document.getElementById('wall-x').addEventListener('input', function(e) {
            document.getElementById('wall-x-value').textContent = e.target.value + ' m';
        });

        document.getElementById('frequency').addEventListener('input', function(e) {
            document.getElementById('freq-value').textContent = e.target.value + ' MHz';
        });

        function updatePlot() {
            const params = {
                tx_x: document.getElementById('tx-x').value,
                tx_y: document.getElementById('tx-y').value,
                rx_x: document.getElementById('rx-x').value,
                rx_y: document.getElementById('rx-y').value,
                wall_x: document.getElementById('wall-x').value,
                wall_height: document.getElementById('wall-height').value,
                freq_MHz: document.getElementById('frequency').value
            };
            
            const queryString = new URLSearchParams(params).toString();
            const plotUrl = '/plot?' + queryString;
            
            document.getElementById('plot-image').src = plotUrl;
            
            // Update results
            fetch('/calculate?' + queryString)
                .then(response => response.json())
                .then(data => {
                    if (data.valid) {
                        document.getElementById('results-text').innerHTML = `
                            <strong>Diffraction Analysis:</strong><br>
                            TX: (${params.tx_x}, ${params.tx_y}) m<br>
                            RX: (${params.rx_x}, ${params.rx_y}) m<br>
                            Wall: x=${params.wall_x}m, h=${params.wall_height}m<br>
                            Frequency: ${params.freq_MHz} MHz<br><br>
                            <strong>Results:</strong><br>
                            Clearance (h): ${data.h.toFixed(2)} m<br>
                            Fresnel param (v): ${data.v.toFixed(2)}<br>
                            Diffraction loss: ${data.diffraction_loss_dB.toFixed(2)} dB<br>
                            FSPL: ${data.fspl_dB.toFixed(2)} dB<br>
                            Total loss: ${data.total_loss_dB.toFixed(2)} dB<br>
                            F1 radius: ${data.f1_radius.toFixed(2)} m
                        `;
                    } else {
                        document.getElementById('results-text').innerHTML = '<strong>Invalid geometry!</strong>';
                    }
                });
        }

        function resetValues() {
            document.getElementById('tx-x').value = 100;
            document.getElementById('tx-y').value = 40;
            document.getElementById('rx-x').value = 400;
            document.getElementById('rx-y').value = 40;
            document.getElementById('wall-x').value = 250;
            document.getElementById('wall-height').value = 35;
            document.getElementById('frequency').value = 900;
            
            // Update displays
            document.getElementById('wall-height-value').textContent = '35 m';
            document.getElementById('wall-x-value').textContent = '250 m';
            document.getElementById('freq-value').textContent = '900 MHz';
            document.getElementById('wall-progress').style.width = '43.75%';
            
            updatePlot();
        }

        // Auto-update on slider changes
        document.getElementById('wall-height').addEventListener('change', updatePlot);
        document.getElementById('wall-x').addEventListener('change', updatePlot);
        document.getElementById('frequency').addEventListener('change', updatePlot);
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path.startswith('/plot'):
            # Parse parameters
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            
            tx_x = float(params.get('tx_x', [100])[0])
            tx_y = float(params.get('tx_y', [40])[0])
            rx_x = float(params.get('rx_x', [400])[0])
            rx_y = float(params.get('rx_y', [40])[0])
            wall_x = float(params.get('wall_x', [250])[0])
            wall_height = float(params.get('wall_height', [35])[0])
            freq_MHz = float(params.get('freq_MHz', [900])[0])
            
            # Generate plot
            plot_data = self.generate_plot(tx_x, tx_y, rx_x, rx_y, wall_x, wall_height, freq_MHz)
            
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()
            self.wfile.write(plot_data)
            
        elif self.path.startswith('/calculate'):
            # Parse parameters
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            
            tx_x = float(params.get('tx_x', [100])[0])
            tx_y = float(params.get('tx_y', [40])[0])
            rx_x = float(params.get('rx_x', [400])[0])
            rx_y = float(params.get('rx_y', [40])[0])
            wall_x = float(params.get('wall_x', [250])[0])
            wall_height = float(params.get('wall_height', [35])[0])
            freq_MHz = float(params.get('freq_MHz', [900])[0])
            
            # Calculate diffraction
            result = self.calc.calculate_diffraction_loss(
                tx_x, tx_y, rx_x, rx_y, wall_x, wall_height, freq_MHz
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    
    def generate_plot(self, tx_x, tx_y, rx_x, rx_y, wall_x, wall_height, freq_MHz):
        """Generate the diffraction plot."""
        # Create figure with vertical layout - both plots large and separate
        fig = plt.figure(figsize=(16, 14))
        
        # Create two separate large subplots vertically
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        
        # ==== MAIN PLOT - LARGE AND SEPARATE ====
        ax1.set_xlim(0, 500)
        ax1.set_ylim(0, 120)
        ax1.set_aspect('equal')
        
        # Draw ground
        ground_y = 20
        ax1.fill_between([0, 500], [0, 0], [ground_y, ground_y], 
                        color='#e2e8f0', alpha=0.5, label='Ground')
        
        # Draw wall
        wall_color = '#475569'
        ax1.add_patch(patches.Rectangle(
            (wall_x - 2, ground_y), 4, wall_height - ground_y,
            facecolor=wall_color, edgecolor='black', linewidth=2
        ))
        
        # Draw line of sight
        ax1.plot([tx_x, rx_x], [tx_y, rx_y], 
                color='#ef4444', linewidth=2, linestyle='--', alpha=0.7,
                label='Line of Sight')
        
        # Draw diffraction path
        ax1.plot([tx_x, wall_x, rx_x], [tx_y, wall_height, rx_y], 
                color='#0ea5e9', linewidth=3, alpha=0.8,
                label='Diffraction Path')
        
        # Draw transmitter and receiver
        ax1.plot(tx_x, tx_y, 'o', color='#1d4ed8', markersize=18, label='TX')
        ax1.plot(rx_x, rx_y, 'o', color='#be123c', markersize=18, label='RX')
        
        # Add labels
        ax1.text(tx_x, tx_y + 8, 'TX', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
        ax1.text(rx_x, rx_y + 8, 'RX', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
        
        # Set labels and title
        ax1.set_xlabel('Distance (m)', fontsize=16)
        ax1.set_ylabel('Height (m)', fontsize=16)
        ax1.set_title('Diffraction Loss Scenario', fontsize=18, fontweight='bold')
        
        # Position legend in upper left corner of plot area
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add calculation results as text on the plot
        result = self.calc.calculate_diffraction_loss(
            tx_x, tx_y, rx_x, rx_y, wall_x, wall_height, freq_MHz
        )
        
        if result['valid']:
            results_text = f"""Results:
v = {result['v']:.2f}
Diff Loss = {result['diffraction_loss_dB']:.1f} dB
FSPL = {result['fspl_dB']:.1f} dB
Total = {result['total_loss_dB']:.1f} dB"""
            
            ax1.text(0.98, 0.98, results_text, transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
        
        # ==== HEATMAP - LARGE AND SEPARATE ====
        x_grid, y_grid, loss_grid = self.calc.generate_heatmap(
            tx_x, tx_y, wall_x, wall_height, freq_MHz, resolution=40
        )
        
        # Create custom colormap
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list('diffraction', colors, N=256)
        
        # Plot heatmap - large and detailed
        im = ax2.imshow(loss_grid, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                       origin='lower', cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
        cbar.set_label('Total Loss (dB)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Mark positions - larger markers for better visibility
        ax2.plot(tx_x, tx_y, 'wo', markersize=12, 
                markeredgecolor='black', markeredgewidth=3, label='TX')
        ax2.plot(rx_x, rx_y, 'wo', markersize=12, 
                markeredgecolor='black', markeredgewidth=3, label='RX')
        ax2.axvline(x=wall_x, color='white', linewidth=4, alpha=0.8, label='Wall')
        
        # Labels
        ax2.set_xlabel('Distance (m)', fontsize=16)
        ax2.set_ylabel('Height (m)', fontsize=16)
        ax2.set_title('Diffraction Loss Heatmap', fontsize=18, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Better spacing between plots
        plt.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.08, hspace=0.3)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plot_data = img_buffer.getvalue()
        plt.close()
        
        return plot_data

def main():
    """Start the web server."""
    print("🌐 Starting Web-Based Diffraction Loss Calculator")
    print("=" * 50)
    print("Features:")
    print("• Interactive web interface")
    print("• Real-time plot updates")
    print("• Wall height progress bar")
    print("• Adjustable TX/RX coordinates")
    print("• Live diffraction calculations")
    print("• Heatmap visualization")
    print()
    print("🚀 Starting server...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    server = HTTPServer(('localhost', 8000), DiffractionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        server.server_close()

if __name__ == "__main__":
    main() 