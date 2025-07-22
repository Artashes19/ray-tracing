import numpy as np
from utd_physics import calculate_utd_coefficient

# ----------------------------------------
# Define 3+ test cases (source, wall, edge_idx, observer, expected)
# ----------------------------------------
TEST_CASES = [
    # (source, wall, edge_idx, observer, (phi0_exp_deg, phi_exp_deg))
    ((100, 100), ((100, 120), (50, 120)), 0, (100, 180), (90.0, 270.0)),
    ((50, 100), ((100, 120), (50, 120)), 1, (100, 180), (90.0, 309.81)),
    ((100, 100), ((100, 120), (50, 120)), 1, (100, 180), (21.8, 309.8)),
]

FREQ = 900e6

print("================= Angle Test Suite =================")
for idx, (src, wall, e_idx, obs, (phi0_exp, phi_exp)) in enumerate(TEST_CASES, 1):
    edge = wall[e_idx]
    D, phi0_rad, phi_rad, _ = calculate_utd_coefficient(src, edge, obs, wall, FREQ)
    phi0_deg = np.degrees(phi0_rad)
    phi_deg = np.degrees(phi_rad)
    print(f"Test {idx}:  calculated  φ0={phi0_deg:.2f}°, φ={phi_deg:.2f}°   | expected  φ0≈{phi0_exp:.2f}°, φ≈{phi_exp:.2f}°")
print("====================================================") 