import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
from torchvision.io import read_image
import matplotlib.pyplot as plt
 
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, asdict
from typing import Union, Tuple, Optional, List

# from kaggle_eval import kaggle_async_eval


IMG_TARGET_SIZE = 640
INITIAL_PIXEL_SIZE = 0.25


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

    def copy(self):
        return RadarSample(
                    self.H,
                    self.W,
                    self.x_ant,
                    self.y_ant,
                    self.azimuth,
                    self.freq_MHz,
                    self.input_img,  
                    self.output_img, 
                    self.pixel_size,
                    self.mask,
                    self.ids,
                )

@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    sampling_position : int
    ids: Optional[Tuple[int, int, int, int]] = None

    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'position_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
    

def read_sample(inputs: Union[RadarSampleInputs, dict]):
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    sampling_position = inputs["sampling_position"]
    
    input_img = read_image(input_file).float()
    C, H, W = input_img.shape
    
    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:  # If single channel, remove channel dimension
            output_img = output_img.squeeze(0)
        
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant, azimuth = sampling_positions.loc[int(sampling_position), ["Y", "X", "Azimuth"]]
    
    sample = RadarSample(
        H=H,
        W=W,
        x_ant=x_ant,
        y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=freq_MHz,
        input_img=input_img,
        output_img=output_img,
        pixel_size=INITIAL_PIXEL_SIZE,
        mask=torch.ones((H, W)),
    )

    if 0 > sample.x_ant >= sample.W or 0 > sample.y_ant >= sample.H:
        print(f"Warning: antenna coords out of range. (x_ant={sample.x_ant}, y_ant={sample.y_ant}), (W={sample.W}, H={sample.H}) -> clamping to valid range.")
    
    return sample


def calculate_fspl(
    dist_m,               # distance in meters (torch tensor)
    freq_MHz,             # frequency in MHz
    min_dist_m=0.125,     # clamp distance below this
):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    fspl_db = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

    return fspl_db



@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def _calculate_transmittance_loss_numpy(
    transmittance_matrix,
    x_ant,
    y_ant,
    n_angles=360*128,
    radial_step=1.0,
    max_walls=10
):
    """
    Numba-accelerated function that casts 'n_angles' rays from (x_ant, y_ant).
    On each crossing from positive->zero in transmittance_matrix, we add path-loss
    to sum_loss. If sum_loss exceeds 160, we clip to 160 and stop the ray.
    """
    h, w = transmittance_matrix.shape
    output  = np.zeros((h, w), dtype=np.float32)
    counts  = np.zeros((h, w), dtype=np.float32)

    dtheta = 2.0 * np.pi / n_angles
    max_dist = np.sqrt(w*w + h*h)
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)

    for i in range(int(n_angles)):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]
        sum_loss  = 0.0
        last_val  = None
        wall_count = 0
        r = 0.0

        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t

            px = int(round(x))
            py = int(round(y))

            if px < 0 or px >= w or py < 0 or py >= h:
                # antenna still outside → step forward until we hit the map
                if last_val is None:
                    r += radial_step
                    continue
                # already inside → leave as before
                if last_val > 0:
                    sum_loss += last_val
                    if sum_loss > 160:
                        sum_loss = 160
                break

            val = transmittance_matrix[py, px]
            if last_val is None:
                last_val = val

            # Detect crossing from positive->zero => add last_val
            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    # If exceeding 160, stop the ray
                    if sum_loss > 160:
                        sum_loss = 160
                        break
                    wall_count += 1
                    if wall_count >= max_walls:
                        # fill remainder with sum_loss
                        r_temp = r
                        while r_temp <= max_dist:
                            x_temp = x_ant + r_temp * cos_t
                            y_temp = y_ant + r_temp * sin_t
                            px_temp = int(round(x_temp))
                            py_temp = int(round(y_temp))
                            if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                                break
                            # average sum_loss into that pixel
                            if counts[py_temp, px_temp] == 0:
                                output[py_temp, px_temp] = sum_loss
                                counts[py_temp, px_temp] = 1
                            else:
                                old_val = output[py_temp, px_temp]
                                old_count = counts[py_temp, px_temp]
                                output[py_temp, px_temp] = (old_val*old_count + sum_loss) / (old_count+1)
                                counts[py_temp, px_temp] += 1
                            r_temp += radial_step
                        break
                last_val = val

            # Average current sum_loss into (px, py)
            if counts[py, px] == 0:
                output[py, px] = sum_loss
                counts[py, px] = 1
            else:
                old_val = output[py, px]
                old_count = counts[py, px]
                output[py, px] = (old_val*old_count + sum_loss) / (old_count+1)
                counts[py, px] += 1

            if wall_count >= max_walls or sum_loss > 160:
                # Check for 160 limit
                if sum_loss > 160:
                    sum_loss = 160
                break

            r += radial_step

    return output




@njit(inline='always')
def _fspl(dist_m, freq_MHz, min_dist_m=0.125):
    """Fast FSPL calculation for numba functions"""
    dist_clamped = max(dist_m, min_dist_m)
    return 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55


@njit(inline='always')
def _step_until_wall(mat, x0, y0, dx, dy, radial_step, max_dist):
    h, w = mat.shape
    x, y = x0, y0
    last_val = mat[int(round(y0)), int(round(x0))]
    travelled = 0.0
    while travelled <= max_dist:
        x += dx * radial_step
        y += dy * radial_step
        travelled += radial_step
        px = int(round(x)); py = int(round(y))
        if px < 0 or px >= w or py < 0 or py >= h:
            return -1, -1, travelled, last_val, last_val
        cur_val = mat[py, px]
        if cur_val != last_val:
            return px, py, travelled, last_val, cur_val
    return -1, -1, travelled, last_val, last_val


@njit(inline='always')
def _estimate_normal(refl_mat, px, py, ray_dx=0.0, ray_dy=0.0):
    """
    Estimate surface normal, handling 1-pixel thick walls properly.
    For thin walls, use ray incident direction to determine correct surface.
    """
    h, w = refl_mat.shape
    val = refl_mat[py, px]
    
    # Check all 4 directions for material boundaries
    left_boundary = px > 0 and refl_mat[py, px-1] != val
    right_boundary = px < w-1 and refl_mat[py, px+1] != val
    top_boundary = py > 0 and refl_mat[py-1, px] != val
    bottom_boundary = py < h-1 and refl_mat[py+1, px] != val
    
    # Standard approach for thick walls
    if left_boundary and not right_boundary:
        return (-1.0, 0.0)  # Wall extends to the right
    elif right_boundary and not left_boundary:
        return (1.0, 0.0)   # Wall extends to the left
    elif top_boundary and not bottom_boundary:
        return (0.0, -1.0)  # Wall extends downward
    elif bottom_boundary and not top_boundary:
        return (0.0, 1.0)   # Wall extends upward
    
    # Handle 1-pixel thick walls (boundaries on both sides)
    elif left_boundary and right_boundary:
        # Horizontal thin wall - use ray direction to pick side
        return (1.0, 0.0) if ray_dx > 0 else (-1.0, 0.0)
    elif top_boundary and bottom_boundary:
        # Vertical thin wall - use ray direction to pick side
        return (0.0, 1.0) if ray_dy > 0 else (0.0, -1.0)
    
    return (0.0, 0.0)  # No clear boundary


@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx
    ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag == 0.0 else (rx/mag, ry/mag)


# ───────────────────────────────────────────────────────────────
#  PAINT-TO-EDGE FALLBACK  (FSPL included)
# ───────────────────────────────────────────────────────────────
@njit(inline='always')
def _paint_to_edge(out_img, x0, y0, dx, dy,
                   acc_loss, path_px,
                   pixel_size, freq_MHz,
                   radial_step, max_dist, max_loss):
    h, w = out_img.shape
    r = 0.0
    while r <= max_dist:
        ix = int(round(x0 + dx*r)); iy = int(round(y0 + dy*r))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return
        fspl = _fspl((path_px + r) * pixel_size, freq_MHz)
        tot  = acc_loss + fspl
        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot if tot < max_loss else max_loss
        r += radial_step


@njit(parallel=True, fastmath=True)
def _calculate_physics_correct_mc_numpy(
    reflectance_matrix: np.ndarray,
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    pixel_size: float,
    freq_MHz: float,
    n_rays: int = 3600,
    max_bounces: int = 5,
    radial_step: float = 0.5,
    max_loss: float = 160.0
) -> np.ndarray:
    """
    Physics-correct Monte Carlo ray tracing with proper wall interactions.
    Uses flag-based approach to handle wall entry/exit properly.
    """
    h, w = reflectance_matrix.shape
    output = np.full((h, w), max_loss, dtype=np.float32)
    
    max_dist = np.hypot(w, h)
    
    # Pre-generate random numbers for reproducibility
    total_rng = n_rays * max_bounces * 2
    rng = np.empty(total_rng, dtype=np.float32)
    seed = 42
    for i in range(total_rng):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        rng[i] = seed / float(2**31)
    
    # Distribute rays uniformly in all directions
    angle_step = 2.0 * np.pi / n_rays
    
    for ray_idx in prange(n_rays):
        # Initial ray direction
        initial_angle = ray_idx * angle_step
        dx = np.cos(initial_angle)
        dy = np.sin(initial_angle)
        
        # Ray state
        x, y = x_ant, y_ant
        cumulative_loss = 0.0
        bounces = 0
        path_distance = 0.0
        
        # Wall state tracking
        inside_wall = False
        wall_entry_trans_val = 0.0
        
        rng_base = ray_idx * max_bounces * 2
        rng_offset = 0
        
        # March ray until termination
        while bounces < max_bounces and cumulative_loss < max_loss:
            # Step until next wall
            hit_px, hit_py, travelled, last_val, cur_val = _step_until_wall(
                reflectance_matrix, x, y, dx, dy, radial_step, max_dist
            )
            
            path_distance += travelled
            
            # Paint along the path with FSPL
            _paint_to_edge(output, x, y, dx, dy, cumulative_loss, path_distance,
                          pixel_size, freq_MHz, radial_step, travelled, max_loss)
            
            # Check if we hit a wall or left the map
            if hit_px < 0 or hit_py < 0:
                # Ray left the map
                if inside_wall:
                    # Exited wall by leaving map - add transmission loss
                    cumulative_loss += wall_entry_trans_val
                    inside_wall = False
                break
            
            # Get material values at hit point
            hit_refl_val = reflectance_matrix[hit_py, hit_px]
            hit_trans_val = transmittance_matrix[hit_py, hit_px]
            
            if not inside_wall:
                # RAY IS IN AIR - check if we hit a wall
                if hit_refl_val > 0 or hit_trans_val > 0:  # Hit wall material
                    # Make reflection/transmission decision
                    if rng_base + rng_offset >= len(rng):
                        break
                    
                    # Probabilistic decision based on material properties
                    total_interaction = hit_refl_val + hit_trans_val
                    if total_interaction > 0:
                        reflection_prob = hit_refl_val / total_interaction
                    else:
                        reflection_prob = 0.5
                    
                    # Clamp probability
                    reflection_prob = max(0.1, min(0.9, reflection_prob))
                    
                    decision_rand = rng[rng_base + rng_offset]
                    rng_offset += 1
                    
                    if decision_rand < reflection_prob:
                        # REFLECTION: bounce and stay in air
                        cumulative_loss += hit_refl_val
                        bounces += 1
                        
                        # Calculate surface normal and reflect
                        nx, ny = _estimate_normal(reflectance_matrix, hit_px, hit_py, dx, dy)
                        
                        # Apply reflection physics
                        if nx == 0.0 and ny == 0.0:
                            # Fallback: reverse direction
                            dx = -dx
                            dy = -dy
                        else:
                            # Proper reflection
                            dx, dy = _reflect_dir(dx, dy, nx, ny)
                        
                        # Move to air position after reflection (small offset)
                        x = float(hit_px) + 0.6 * dx
                        y = float(hit_py) + 0.6 * dy
                        
                    else:
                        # TRANSMISSION: enter wall
                        inside_wall = True
                        wall_entry_trans_val = hit_trans_val
                        # Direction unchanged, move into wall
                        x = float(hit_px) + 0.1 * dx
                        y = float(hit_py) + 0.1 * dy
                else:
                    # Hit air - just continue
                    x, y = float(hit_px), float(hit_py)
            else:
                # RAY IS INSIDE WALL - check if we exited
                if hit_refl_val == 0 and hit_trans_val == 0:  # Back to air
                    # Exited wall - add transmission loss now
                    cumulative_loss += wall_entry_trans_val
                    inside_wall = False
                    wall_entry_trans_val = 0.0
                    # Continue in air
                    x, y = float(hit_px), float(hit_py)
                else:
                    # Still inside wall - keep marching
                    x, y = float(hit_px), float(hit_py)
    
    return output


class Approx:
    def approximate(self, sample : RadarSample) -> torch.Tensor:
        ref, trans, dist = sample.input_img.cpu().numpy()
        x_ant, y_ant = sample.x_ant, sample.y_ant

        fspl = calculate_fspl(dist_m=dist, freq_MHz=sample.freq_MHz)

        # PHYSICS-CORRECT: Use the new corrected ray propagation
        physics_correct_loss = _calculate_physics_correct_mc_numpy(
            ref, trans, x_ant, y_ant,
            pixel_size=sample.pixel_size,
            freq_MHz=sample.freq_MHz,
            n_rays=360*32,      # Reasonable number for realistic patterns
            max_bounces=4,      # Allow realistic multi-bounce reflections  
            radial_step=0.5,    # Fine steps for accurate interfaces
            max_loss=160.0      # Maximum loss threshold
        )
        
        # Combine with baseline transmission for robustness
        baseline_trans = _calculate_transmittance_loss_numpy(
            trans, x_ant, y_ant, 
            n_angles=360*64,    # Reduced since we have realistic method
            radial_step=1.0, 
            max_walls=8
        )
        
        # Weight combination: favor physics-correct method but ensure coverage
        alpha = 0.8  # Weight for physics-correct method
        combined_loss = alpha * physics_correct_loss + (1.0 - alpha) * baseline_trans
        
        # Light smoothing and add FSPL
        smoothed = gaussian_filter(combined_loss, sigma=0.8, mode='reflect')
        approx = np.minimum(smoothed + fspl, 160.0)

        return torch.from_numpy(approx)
    
    def predict(self, samples):
        # samples = [read_sample(s) for s in samples]
        predictions = [self.approximate(s) for s in tqdm(samples, "predicting...")]

        return predictions

BASE_DIR = os.path.dirname(__file__)

freqs_MHz = [868]
DATA_DIR = os.path.expanduser("~/data/train/")
INPUT_PATH = os.path.join(DATA_DIR, f"Inputs/Task_1_ICASSP/")
OUTPUT_PATH = os.path.join(DATA_DIR, f"Outputs/Task_1_ICASSP/")
POSITIONS_PATH = os.path.join(DATA_DIR, "Positions/")
BUILDING_DETAILS_PATH = os.path.join(DATA_DIR, "Building_Details/")
RADIATION_PATTERNS_PATH = os.path.join(DATA_DIR, "Radiation_Patterns/")

inputs_list = []
for b in range(1, 26):  # 25 buildings
    for ant in range(1, 3):  # 2 antenna types
        for f in range(1, 2):  # 3 frequencies
            for sp in range(80):  # 80 sampling positions
                # Check if file exists
                input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                
                if os.path.exists(os.path.join(INPUT_PATH, input_file)) and \
                    os.path.exists(os.path.join(OUTPUT_PATH, output_file)):
                    input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    radiation_file = f"Ant{ant}_Pattern.csv"
                    position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"

                    freq_MHz = freqs_MHz[f-1]
                    input_img_path = os.path.join(INPUT_PATH, input_file)
                    output_img_path = os.path.join(OUTPUT_PATH, output_file)
                    positions_path = os.path.join(POSITIONS_PATH, position_file)

                    
                    radar_sample_inputs = RadarSampleInputs(
                        freq_MHz=freq_MHz,
                        input_file=input_img_path,
                        output_file=output_img_path,
                        position_file=positions_path,
                        sampling_position=sp,
                        ids=(b, ant, f, sp),
                    )

                    inputs_list.append(radar_sample_inputs)

train_samples = [read_sample(s) for s in inputs_list]
print(f"Read all {len(train_samples)} Train Samples")

def rmse(pred, true):
    return torch.sqrt(torch.mean(torch.square(pred-true))).item()

predictor = Approx().predict
preds = predictor(samples=train_samples)

errs = []
for i in tqdm(range(len(train_samples))):
    pred, true = preds[i], train_samples[i].output_img
    err = rmse(pred, true)
    errs.append(err)

errs = np.array(errs)
err_order = np.argsort(errs)[::-1]
print(np.mean(errs))
