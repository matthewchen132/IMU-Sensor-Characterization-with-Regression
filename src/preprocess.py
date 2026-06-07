import pandas as pd
import numpy as np
def downsample(df, factor) -> pd.DataFrame:
    """
    Reduces dataset size by taking every `factor`-th row.    
    Bias thermal drift occurs over seconds-to-minutes timescales.
    Millisecond resolution (200Hz) is unnecessary for this regression.
    
    factor=200  →  1Hz,  ~400k samples  (fast)
    factor=2000 →  0.1Hz, ~40k samples  (very fast)
    """
    return df.iloc[::factor].reset_index(drop=True)

def align_gravity(df) -> pd.DataFrame:
    """
    Pre-processing step 1, account for tilting in data-set.

    Our IMU is static but tilted. Gravity leaks into accel_x/y.
    We estimate the sensor orientation from the mean accel vector
    (valid only because the IMU is static over the full sequence),
    then rotate all accelerations into the level frame.
    
    After rotation: truth is [0, 0, 9.81] in body frame.
    Residual after subtracting truth = bias signal for regression.
    """
    # Estimate mean gravity direction from static data
    g_body = np.array([
        df['accel_x'].mean(),
        df['accel_y'].mean(),
        df['accel_z'].mean()
    ])
    g_mag = np.linalg.norm(g_body)

    # Unit vector pointing in gravity direction (body frame)
    g_hat = g_body / g_mag  # should be close to [0, 0, 1] if level

    # Roll and pitch from static accel (standard IMU leveling)
    roll  = np.arctan2(g_hat[1], g_hat[2])
    pitch = np.arctan2(-g_hat[0], np.sqrt(g_hat[1]**2 + g_hat[2]**2))

    print(f"Estimated tilt — roll: {np.degrees(roll):.2f}°, "
          f"pitch: {np.degrees(pitch):.2f}°")

    # Rotation matrix: body frame → "flat" frame (Rx * Ry)
    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(roll), -np.sin(roll)],
                   [0,  np.sin(roll),  np.cos(roll)]])

    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0             ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    R = Rx @ Ry  # combined rotation

    # Rotate each accelerometer sample
    accel = df[['accel_x', 'accel_y', 'accel_z']].values  # (N, 3)
    accel_aligned = (R @ accel.T).T  # (N, 3)

    df = df.copy()
    df['accel_x'] = accel_aligned[:, 0]
    df['accel_y'] = accel_aligned[:, 1]
    df['accel_z'] = accel_aligned[:, 2]
    return df

def smooth_by_time_bins(features, labels, n: int) -> pd.DataFrame:
    '''
    Smooths noise by creating bins.
      t (seconds)
    '''
    current_i = 0
    features = features.copy()
    labels = labels.copy()

    feat_bins = []
    label_bins = []
    while(current_i + n < len(features)):
        feat_bins.append(features.iloc[current_i:current_i+n].mean())
        label_bins.append(labels.iloc[current_i:current_i+n].mean())
        current_i += n
    
    features_binned = pd.DataFrame(feat_bins).reset_index(drop=True)
    labels_binned   = pd.DataFrame(label_bins).reset_index(drop=True)

    print(f"[smooth] {len(features):,} rows → {len(features_binned):,} bins  "
          f"({n} rows/bin)  "
          f"noise reduction ≈ {np.sqrt(n):.1f}x")

    return features_binned, labels_binned