# Project Log: 2D Ray-Tracing Simulator Enhancement

## Project Overview

This log documents the enhancement of a 2D ray-tracing simulator designed to model radio signal propagation. The baseline simulator already accounts for several key physical phenomena:

*   **Free Space Path Loss (FSPL):** Signal attenuation over distance in open air.
*   **Transmittance Loss:** Signal reduction when passing through objects like walls.
*   **Reflectance Loss:** Signal reduction upon bouncing off surfaces.

The primary goal of the work detailed here is to implement a critical missing component: **electromagnetic diffraction**. This involves calculating the energy loss that occurs when a signal bends around the sharp edge of an obstacle. The effort is focused on developing and verifying the core diffraction physics in a standalone module (`utd_physics.py`) before integrating it into the larger ray-tracing application.

Below is a dated changelog recording major decisions, fixes, and milestones for this diffraction implementation effort.

---

### 2025-07-22

**Objective: Stabilize and verify the core angle calculations in `utd_physics.py`**

*   **Repository Cleanup:**
    *   Deleted legacy and unused files (`gtd_visualizer.py`, `approx.py`) to focus the workspace on the core physics (`utd_physics.py`) and its test harness (`angle_visualizer.py`).

*   **Bug Fix: Initial `TypeError` in Visualizer**
    *   **Problem:** `angle_visualizer.py` was crashing because `calculate_utd_coefficient` returned a single complex number, while the visualizer expected three values (coefficient, phi_0, phi).
    *   > **Fix:** Modified `utd_physics.py` to return `(D_complex, phi_0_rad, phi_rad)`, resolving the unpacking error.

*   **Bug Fix: Remote Plotting Failure**
    *   **Problem:** `plt.show()` in `angle_visualizer.py` failed to display a GUI on the remote server environment.
    *   > **Fix:** Changed the output to `plt.savefig('diffraction_plot.png')` to generate a static image file, which is a robust solution for remote development.

*   **Project Brief & Theory Review:**
    *   **Decision:** Officially adopted the Uniform Theory of Diffraction (UTD) as the target physical model, upgrading from the simpler but flawed Geometrical Theory of Diffraction (GTD).
    *   **Action:** Reviewed and summarized key papers (Keller's 1962 GTD, WRDC UTD Report) to establish the correct formulas for the diffraction coefficient `D`, the wedge parameters (`beta`, `n`), and the Fresnel transition function `F(tau)`.

*   **Feature: GTD Coefficient Implementation**
    *   **Decision:** To simplify the ray-tracing implementation, we will use a pure GTD coefficient and avoid launching rays near the known shadow boundaries where it becomes singular.
    *   **Action:** Implemented the user-provided secant-based formula for the GTD coefficient in `utd_physics.py`.

*   **Bug Fix: Critical Angle Calculation Logic**
    *   **Problem:** The existing logic for determining the reference "0-face" and calculating `phi_0` and `phi` was ambiguous and failed on several geometric test cases provided by the user.
    *   > **Fix:** Rewrote the angle calculation logic from scratch. The final, robust implementation uses a single, clear rule: **All angles (`phi_0`, `phi`) are measured clockwise from the physical wall vector (defined as pointing from the diffracting edge towards the other point of the wall).** This new logic successfully passed all provided test cases.

*   **Feature: Automated Test Suite**
    *   **Action:** Created a new script, `test_utd_angles.py`, to serve as a regression test suite.
    *   > **Content:** The script codifies all known test cases and runs them automatically, printing the calculated vs. expected angles for rapid verification of `utd_physics.py`. 