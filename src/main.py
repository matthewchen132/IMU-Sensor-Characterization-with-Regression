from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import helpers
import numpy as np
import pandas as pd
import allantools  # pip install allantools
from helpers import import_data
from sklearn.linear_model import RidgeCV, LassoCV

G_TRUE      = 9.80665
# gyro_data: rate samples [rad/s], dt: sample period [s]
def run_ridge(df, state):
    '''
    Runs ridge regression on our dataset to extract our most valuable terms
    
    returns:
     - np array with metrics on contribution amount
    '''
def run_lasso(df, state):
    '''
    Runs ridge regression on our dataset to extract our most valuable terms
    
    returns:
     - np array with metrics on contribution amount
    '''
def compare_ridge_lasso():
    '''
    compare results of bias weighting compared to allantools results from dataset.
    
    metrics might include mean squared error, mean absolute error, or coefficient of de-
    termination (R2).
    '''
    # rate, adev, adev_err, _ = allantools.oadev(gyro_data, rate=1/dt, data_type='freq')
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

def generate_features(df) -> pd.DataFrame:
    '''
    Generates feature vector for ridge and lasso regression.
    Features are motivated by MEMS IMU thermal/temporal bias models.
    [1, T, T², T_dot, t]    
    '''
    # y = measured - [0, 0, 0, 0, 0, 9.81]
    ones = np.ones(len(df))            # Static bias offset (DC-turn on, etc.)
    t = df['timestamp'].values
    t =  t - t[0]
    T = df['temperature']
    T_2 = T ** 2
    T_dot = np.gradient(T,t)

    features = pd.DataFrame({
            'bias_const': ones,   # DC / turn-on bias
            'T':          T,                   # linear thermal
            'T_2':        T ** 2,              # nonlinear thermal (MEMS datasheets)
            'T_dot':      T_dot,               # thermal gradient stress
            't':          t,                   # in-run drift after thermal terms
        })
    return features

def generate_labels(df) -> pd.DataFrame:
    '''
    Generates labels for our processed data based on the error from stationary value
    stationary IMU reads: [.0, .0, .0, .0, .0, 9.81]
    '''
    labels = pd.DataFrame({
        "bias_gyro_x" : df['gyro_x'].values - 0.0,
        "bias_gyro_y" : df['gyro_y'].values - 0.0,
        "bias_gyro_z" : df['gyro_z'].values - 0.0,
        "bias_ax" : df['accel_x'].values    - 0.0,
        "bias_ay" : df['accel_y'].values    - 0.0,
        "bias_az" : df['accel_z'].values    - G_TRUE, # Gravity offset
    })

    return labels

def run_regression(X,y) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit here
    X_test  = scaler.transform(X_test)        # transform only — no refit
    lasso_pipeline = make_pipeline(
        StandardScaler(),
        LassoCV(cv=5, random_state=42), 
        shuffle=True
    )

    lasso_pipeline.fit(X_train, y_train)
    best_lasso = lasso_pipeline.named_steps['lassocv']



def main():
    unprocessed_df = import_data(file='dataset-calib-imu-static2.npy') # Columns: timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, temperature

    # 1) Pre-filter tilt during calibration.
    processed_df = align_gravity(unprocessed_df)
    features = generate_features(processed_df)
    labels = generate_labels(processed_df)
    run_regression(X=features, y=labels)


    # rate, adev, adev_err, _ = allantools.oadev(gyro_data, rate=1/dt, data_type='freq')




if __name__ == "__main__":
    main()
    