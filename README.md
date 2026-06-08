# IMU Sensor Characterization with Regression
Comparing Ridge and Lasso Regression for Long-term Bias Estimation from Time-stamped IMU & Temperature Data

## Setup

```bash
git clone https://github.com/matthewchen132/IMU-Sensor-Characterization-with-Regression
cd IMU-Sensor-Characterization-with-Regression
conda create -n <env_name> python=3.10
conda activate <env_name>
pip3 install -r requirements.txt
```

## Download Dataset

The dataset (~4.8 GB) is from the [TUM Visual-Inertial Dataset](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) and must be downloaded separately:

```bash
curl -L -o dataset-calib-imu-static2.npy \
  https://cdn2.vision.in.tum.de/tumvi/imu_static/dataset-calib-imu-static2.npy
```

Place the file in the root of the repository.

## Run

```bash
python src/main.py
```

## Repository Structure

```
.
├── src/          # Source code
├── Figures/      # Output plots: Allan variance, Lasso/Ridge terms, regularization constants (lambda)
├── requirements.txt
└── dataset-calib-imu-static2.npy   # (not tracked — download separately)
```

## Bias Plot
<img width="1800" height="2250" alt="image" src="https://github.com/user-attachments/assets/f6cf4030-f02e-4c0e-8b12-424ca15b7e32" />

## Correction to Report / Ridge + Lasso Result
Ridge vs. Lasso: Our Ridge and Lasso regression performed about equally, but we saw Lasso zero out all terms except √T and Ṫ. The similar performance suggests that the constructed feature set is pretty well-constructed, and the regularization is mainly preventing overfitting rather than doing model selection. However, the zeroed out terms signify that √T and Ṫ are most significant in the bias modeling of the IMU. 
<img width="375" height="674" alt="image" src="https://github.com/user-attachments/assets/0e6b4518-6900-4111-9f10-6b144eeb6028" />

Fig. 6: Lasso regression shows reduction of non-contributing terms to 0, while ridge regression keeps some smaller weights on the weaker contributing terms. Contributing terms (√T, Ṫ) have notably higher weight values.
