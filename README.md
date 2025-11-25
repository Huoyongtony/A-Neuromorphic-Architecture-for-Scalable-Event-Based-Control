# A-Neuromorphic-Architecture-for-Scalable-Event-Based-Control


## How to reproduce the figures
1. Install the Python dependencies used in the legacy notebooks (numpy, numba, matplotlib, mujoco, gym, pyvis, imageio, networkx, webcolors).
2. Launch Jupyter from the repository root so the `clean` package is importable.
3. Open any notebook in `clean/notebooks/` and run all cells:
   - `cell.ipynb` writes figures to `figures/cell/`.
   - `ring.ipynb` writes to `clean/notebooks/figures/ring/`.
   - Snake notebooks write to their own folders (`animations_only_HCO_2`, `snake_pattern`, `snake_pattern_supervisory`).
4. For the snake notebooks, keep `MUJOCO_GL=egl` (already set in the first cell) and ensure your MuJoCo install can load the provided XMLs.

## Folder map
- `legacy/` – original notebooks and utilities.
- `clean/lib/` – reusable code for dynamics, graphs, and the MuJoCo snake environment.
- `clean/notebooks/` – unified notebooks that call the new libraries while keeping plotting behaviour identical.
