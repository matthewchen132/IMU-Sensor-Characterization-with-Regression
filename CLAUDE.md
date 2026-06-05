# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Compare Ridge and Lasso regression for characterizing MEMS IMU sensor biases (accelerometer + gyroscope) as functions of temperature and time. The dataset is a static calibration sequence where the sensor is held still, so any measured signal beyond [0, 0, 0, 0, 0, 9.80665] is bias.

## Running the Code

The dataset (`dataset-calib-imu-static2.npy`) lives in the repo root, but `main.py` uses a bare relative import (`import helpers`) and loads the `.npy` by relative path. Run from `src/` with the repo root on the path:

```bash
cd src && PYTHONPATH=. python main.py ../dataset-calib-imu-static2.npy
```

Or from repo root:

```bash
PYTHONPATH=src python src/main.py
```

Install dependencies (conda environment preferred per `.vscode/settings.json`):

```bash
pip install numpy pandas scikit-learn allantools
```

## Architecture

```
src/
  helpers.py   — data loading: imports .npy, returns DataFrame with 8 columns
  main.py      — full pipeline: align_gravity → generate_features → generate_labels → run_regression
```

**Pipeline stages in `main.py`:**

1. `align_gravity(df)` — estimates sensor tilt from mean static accel vector, computes roll/pitch, rotates all accel samples into the level frame so the truth is exactly [0, 0, g].
2. `generate_features(df)` — builds the regressor matrix `[1, T, T², T_dot, t]` motivated by MEMS thermal bias models (DC offset, linear/nonlinear thermal, thermal gradient stress, in-run drift).
3. `generate_labels(df)` — residuals: `measured − truth` for all 6 DOF (gyro truth = 0 rad/s, accel_z truth = 9.80665 m/s²).
4. `run_regression(X, y)` — train/test split → StandardScaler → RidgeCV / LassoCV with 5-fold CV.

**Dataset columns:** `timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, temperature`  
Timestamps are nanoseconds. `allantools.oadev` is imported for Allan deviation analysis of gyro noise but not yet wired in.

**Stubs to implement:** `run_ridge`, `run_lasso`, `compare_ridge_lasso` in `main.py` are empty. `src/ridge_regression` is an empty placeholder file.

## Key Constants

- `G_TRUE = 9.80665` m/s² — reference gravity used for accel_z label
- Feature `T_dot` is computed via `np.gradient(T, t)` (temperature rate of change)

## Notes: 