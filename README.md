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
