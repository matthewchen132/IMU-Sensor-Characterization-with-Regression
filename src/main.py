from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import preprocess as pre
import numpy as np
import pandas as pd
import allantools
import matplotlib.pyplot as plt
from helpers import import_data
from sklearn.linear_model import RidgeCV, LassoCV
import time
G_TRUE = 9.80665

def generate_features(df) -> pd.DataFrame:
    '''
    Feature vector motivated by MEMS IMU thermal/temporal bias models.
    [bias_const, T_sqrt, T, T², T_dot, t]
    '''
    ones = np.ones(len(df)) # bias_const
    t = df['timestamp'].values
    t = t - t[0]
    T = df['temperature']
    T_dot = np.gradient(T, t)

    return pd.DataFrame({
        'bias_const': ones,
        'T_sqrt': T ** 0.5,
        'T': T,
        'T_2': T ** 2,
        'T_dot':      T_dot,
        't':          t,
    })

def generate_labels(df) -> pd.DataFrame:
    '''
    Bias = deviation from ideal static IMU reading.
    Gyro ideal: 0 rad/s each axis. Accel ideal: [0, 0, 9.81] m/s².
    '''
    return pd.DataFrame({
        "bias_gyro_x": df['gyro_x'].values,
        "bias_gyro_y": df['gyro_y'].values,
        "bias_gyro_z": df['gyro_z'].values,
        "bias_ax":     df['accel_x'].values,
        "bias_ay":     df['accel_y'].values,
        "bias_az":     df['accel_z'].values - G_TRUE,
    })

def run_regression(X, y) -> tuple[dict, StandardScaler]:
    # split: train on first 80%, test on last 20%. (train_test_split)
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    X_all   = scaler.transform(X)

    results = {}
    n_axes  = len(y.columns)

    for i, col in enumerate(y.columns):
        print(f"\n[{i+1}/{n_axes}] Fitting axis: {col}")
        y_train_col = y_train[col].values
        y_test_col  = y_test[col].values

        # Ridge with 5-fold CV
        ridge = RidgeCV(
            alphas=np.logspace(-8, 6, 200),
            cv=5, scoring='neg_mean_squared_error'
        ).fit(X_train, y_train_col)
        print(f"Ridge done | best λ={ridge.alpha_}")

        # Lasso with 5-fold CV
        lasso = LassoCV(
            alphas=np.logspace(-8, 2, 200),
            cv=5, max_iter=50000,
            tol=1e-4, random_state=42,
        ).fit(X_train, y_train_col)
        print(f"Lasso done | best λ={lasso.alpha_}")

        results[col] = {}
        for name, model in [('Ridge', ridge), ('Lasso', lasso)]:
            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)
            n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
            results[col][name] = {
                'model': model, 'alpha': model.alpha_,
                'coefs': model.coef_, 'intercept': model.intercept_,
                'train_mse': mean_squared_error(y_train_col, y_pred_train), 'train_r2': r2_score(y_train_col, y_pred_train),
                'mse': mean_squared_error(y_test_col, y_pred_test), 'mae': mean_absolute_error(y_test_col, y_pred_test), 
                'r2': r2_score(y_test_col, y_pred_test), 'n_nonzero': n_nonzero, 'y_pred_all': model.predict(X_all),
            }
            print(f"  [{col}] {name}  lambda={model.alpha_}  "
                  f"nonzero={n_nonzero}/{len(X.columns)}  "
                  f"train_R²={results[col][name]['train_r2']}  "
                  f"test_R²={results[col][name]['r2']}")
    return results, scaler

def print_metrics(results):
    header = f"{'Axis'} {'Model'} {'Train MSE'} {'Test MSE'} {'Test MAE'} {'Train R²'} {'Test R²'}"
    print("\n" + header)
    print("-" * len(header))
    for col, models in results.items():
        for name, r in models.items():
            print(f"{col} {name} "
                  f"{r['train_mse']} "
                  f"{r['mse']} "
                  f"{r['mae']} "
                  f"{r['train_r2']} "
                  f"{r['r2']}")

def print_physical_biases(results, scaler, feature_names):
    '''
    Previously, we used the Standard Scaler to make factors similar in size (prevents ill-conditioning)
    Unscales coefficients from normalized to physical units.
    y = coef_norm @ X_norm + intercept_norm
      = (coef_norm/scale) @ X + (intercept_norm - (coef_norm/scale)·mean)
    '''
    header = f"\n{'Axis'} {'Model':<7} {'DC_bias'}  " + \
             "  ".join(f"{n:>12}" for n in feature_names)
    print(header)
    print("-" * (15 + 7 + 14 + 14 * len(feature_names)))
    for col, models in results.items():
        for name, r in models.items():
            coef_phys = r['coefs'] / scaler.scale_
            dc_bias   = r['intercept'] - np.dot(r['coefs'] / scaler.scale_, scaler.mean_)
            row = f"{col} {name} {dc_bias}  " + \
                  "  ".join(f"{c}" for c in coef_phys)
            print(row)

def plot_regression_results(results, features, labels, scaler):
    t_hours = features['t'].values / 3600.0
    feature_names = list(features.columns)
    axes = list(results.keys())
    n_axes = len(axes)

    # --- Figure 1: CV alpha selection curves ---
    fig, axs = plt.subplots(n_axes, 2, figsize=(12, 2.5 * n_axes), sharex='col')
    for i, col in enumerate(axes):
        # Ridge: 5-fold CV doesn't expose per-alpha scores; show selected alpha only
        ridge_model = results[col]['Ridge']['model']
        ax = axs[i, 0]
        ax.axvline(ridge_model.alpha_, color='r', linestyle='--')
        ax.set_xscale('log')
        ax.text(0.5, 0.5, f'Selected λ* = {ridge_model.alpha_:.2e}',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_ylabel(col, fontsize=8)
        if i == 0:
            ax.set_title('Ridge: 5-Fold CV Selected λ')

        lasso_model = results[col]['Lasso']['model']
        lasso_cv_mse = lasso_model.mse_path_.mean(axis=1)
        ax = axs[i, 1]
        ax.semilogx(lasso_model.alphas_, lasso_cv_mse)
        ax.axvline(lasso_model.alpha_, color='r', linestyle='--',
                   label=f'λ*={lasso_model.alpha_:.1e}')
        if i == 0:
            ax.set_title('Lasso: 5-Fold CV MSE vs λ')
        ax.legend(fontsize=7)

    axs[-1, 0].set_xlabel('lambda (regularization strength)')
    axs[-1, 1].set_xlabel('lambda (regularization strength)')
    fig.suptitle('Cross-Validation: Regularization Parameter Selection', fontweight='bold')
    fig.tight_layout()
    plt.savefig('cv_curves.png')

    # --- Figure 2: Physical coefficients Ridge vs Lasso ---
    x = np.arange(len(feature_names))
    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 3 * n_axes))
    if n_axes == 1:
        axs = [axs]
    for i, col in enumerate(axes):
        ridge_c = results[col]['Ridge']['coefs'] / scaler.scale_
        lasso_c = results[col]['Lasso']['coefs'] / scaler.scale_
        axs[i].bar(x - 0.2, ridge_c, 0.4, label='Ridge', color='steelblue')
        axs[i].bar(x + 0.2, lasso_c, 0.4, label='Lasso', color='darkorange')
        axs[i].axhline(0, color='k', linewidth=0.5)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(feature_names, rotation=30, ha='right')
        axs[i].set_ylabel(col, fontsize=8)
        axs[i].legend(fontsize=8)
    axs[0].set_title('Physical Feature Coefficients: Ridge vs Lasso')
    fig.tight_layout()
    plt.savefig('coefficients.png')

    # --- Figure 3: Predicted bias vs time ---
    fig, axs = plt.subplots(n_axes, 1, figsize=(12, 2.5 * n_axes), sharex=True)
    if n_axes == 1:
        axs = [axs]
    for i, col in enumerate(axes):
        axs[i].plot(t_hours, labels[col].values, 'k.', ms=2, alpha=0.4, label='Measured')
        axs[i].plot(t_hours, results[col]['Ridge']['y_pred_all'], 'b-', lw=1.5, label='Ridge')
        axs[i].plot(t_hours, results[col]['Lasso']['y_pred_all'], 'g--', lw=1.5, label='Lasso')
        axs[i].set_ylabel(col, fontsize=8)
        axs[i].legend(fontsize=7, loc='upper right')
    axs[-1].set_xlabel('Time (hours)')
    fig.suptitle('IMU Bias: Measured vs Ridge/Lasso Model', fontweight='bold')
    fig.tight_layout()
    plt.savefig('bias_timeseries.png')

    # --- Figure 4: Train vs Test R² comparison ---
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_axes)
    w = 0.2
    configs = [
        ('Ridge', 'train_r2', 'steelblue',     'Ridge train'),
        ('Ridge', 'r2',       'cornflowerblue', 'Ridge test'),
        ('Lasso', 'train_r2', 'darkorange',     'Lasso train'),
        ('Lasso', 'r2',       'moccasin',       'Lasso test'),
    ]
    for j, (model_name, key, color, label) in enumerate(configs):
        vals = [results[col][model_name][key] for col in axes]
        ax.bar(x + (j - 1.5) * w, vals, w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(axes, rotation=30, ha='right')
    ax.set_ylabel('R²')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_title('Train vs Test R²: Ridge and Lasso')
    fig.tight_layout()
    plt.savefig('train_test_r2.png')
    plt.show()
    print("\nSaved: cv_curves.png  coefficients.png  bias_timeseries.png  train_test_r2.png")

def plot_allan_deviation(df_1hz):
    '''
    Allan deviation characterizes IMU noise independently of regression.
    '''
    rate = 1.0  # Hz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for col in ['gyro_x', 'gyro_y', 'gyro_z']:
        taus, ad, _, _ = allantools.adev(df_1hz[col].values, rate=rate,
                                         data_type='freq', taus='all')
        ax1.loglog(taus, ad, label=col)
    ax1.set_xlabel('Averaging time τ (s)')
    ax1.set_ylabel('Allan Deviation (rad/s)')
    ax1.set_title('Gyroscope Allan Deviation')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    for col in ['accel_x', 'accel_y', 'accel_z']:
        taus, ad, _, _ = allantools.adev(df_1hz[col].values, rate=rate,
                                         data_type='freq', taus='all')
        ax2.loglog(taus, ad, label=col)
    ax2.set_xlabel('Averaging time τ (s)')
    ax2.set_ylabel('Allan Deviation (m/s²)')
    ax2.set_title('Accelerometer Allan Deviation')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    plt.savefig('allan_deviation.png')
    plt.show()
    print("Saved: allan_deviation.png")

def main():
    raw_df = import_data(file='dataset-calib-imu-static2.npy')

    # -- Pre-process --
    # Correct for sensor tilt (accel from gravity leaks into accel_x/y when not perfectly level)
    aligned_df = pre.align_gravity(raw_df)
    # Downsample 200 Hz → 1 Hz (bias drift is on the seconds-to-minutes scale)
    df_1hz = pre.downsample(aligned_df, factor=200)

    # -- Build features (X) / Labels (y) --
    # Allan deviation — noise floor characterization independent of regression
    plot_allan_deviation(df_1hz)
    # Build features and labels on the 1 Hz data
    features = generate_features(df_1hz)
    labels = generate_labels(df_1hz)


    # Bin-average into 30-second windows (~5.47x noise reduction, Final pre-processing step)
    features, labels = pre.smooth_by_time_bins(features=features, labels=labels, n=30)
    # Ridge + Lasso (5-fold CV) with train/test split
    results, scaler = run_regression(X=features, y=labels)

    # -- Results --
    # Print MSE / MAE / R² for all axes and models
    print_metrics(results)
    # Convert weights to physical units and print interpretation table
    print_physical_biases(results, scaler, list(features.columns))
    # Visualize CV curves, coefficients, bias time series, train vs test R²
    plot_regression_results(results, features, labels, scaler)


if __name__ == "__main__":
    main()
