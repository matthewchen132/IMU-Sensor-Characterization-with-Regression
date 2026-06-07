import numpy as np
import pandas as pd

def import_data(file):
    data = np.load(file)
    print(data.shape)   # Should be (rows, 8)
    print(data[:3])     # Preview first 3 rows

    # Columns: timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, temperature
    df = pd.DataFrame(data, columns=[
        'timestamp', 'gyro_x', 'gyro_y', 'gyro_z',
        'accel_x', 'accel_y', 'accel_z', 'temperature'
    ])
    return df

# These readings are static (a = 9.81 in z-dir, and gyro = 0.0 rads/s)


# -- Ridge / Lasso states --
# 1 : Constant bias due to DC-turn on, etc.
# t, t^2, t^3, ; Potential Time dependence
# T, T^2, T^3, ; Relation to absolute temperature
# T_dot, T_dot2; Relation to temperature ROC
# a_mag ; error in the true state value
# gyrr_mag ; error in the true state value
