import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
WINDOW_SIZE = 100       # samples (100Hz → 1 second)
WINDOW_STRIDE = 10      # step between windows
IMAGE_SIZE = 150        # for DNN/VGG16 input (150x150)

# Joint moments to predict (from inverse_dynamics.csv)
MOMENT_COLS = [
    'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment',
    'hip_flexion_l_moment', 'hip_adduction_l_moment', 'hip_rotation_l_moment',
    'knee_angle_r_moment', 'knee_angle_l_moment',
    'ankle_angle_r_moment', 'ankle_angle_l_moment'
]

# IMU sensor files to use (lower limb focus)
IMU_SENSORS = [
    'LThigh', 'RThigh', 'LShank', 'RShank', 'LFoot', 'RFoot', 'Pelvis'
]

# Columns per IMU sensor (13 columns, no header)
IMU_COL_NAMES = [
    'acc_x', 'acc_y', 'acc_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z',
    'quat_w', 'quat_x', 'quat_y', 'quat_z'
]

# ─────────────────────────────────────────
# MARYAM DATASET LOADER
# ─────────────────────────────────────────
def load_maryam_subject(imu_subj_path, moment_subj_path, prefix='LLEG-000'):
    """
    Load IMU and joint moment data for one Maryam subject.
    Returns: imu_array (N x n_features), moments_array (N x n_moments)
    """
    # Load IMU sensors
    imu_data = []
    for sensor in IMU_SENSORS:
        # Try both LLEG and RLEG prefixes
        for pfx in ['LLEG-000', 'RLEG-000']:
            fpath = os.path.join(imu_subj_path, 'IMU', '100Hz', 'FC', 'LEG',
                                 f'{pfx}_{sensor}.txt')
            if os.path.exists(fpath):
                df = pd.read_csv(fpath, header=None, names=IMU_COL_NAMES)
                imu_data.append(df.values)
                break

    if len(imu_data) == 0:
        print(f"No IMU data found at {imu_subj_path}")
        return None, None

    # Stack all sensors horizontally
    imu_array = np.hstack(imu_data)  # (N, n_sensors * 13)

    # Load joint moments
    moment_path = os.path.join(moment_subj_path, 'ID',
                               'result_lifting_weight', 'inverse_dynamics.csv')
    if not os.path.exists(moment_path):
        print(f"No moment file found at {moment_path}")
        return None, None

    moments_df = pd.read_csv(moment_path)

    # Keep only the moment columns we want
    available = [c for c in MOMENT_COLS if c in moments_df.columns]
    moments_array = moments_df[available].values  # (M, n_moments)

    # Align lengths (resample moments to match IMU length)
    n_imu = imu_array.shape[0]
    n_mom = moments_array.shape[0]

    if n_imu != n_mom:
        # Resample moments to IMU length
        x_old = np.linspace(0, 1, n_mom)
        x_new = np.linspace(0, 1, n_imu)
        resampled = np.zeros((n_imu, moments_array.shape[1]))
        for i in range(moments_array.shape[1]):
            f = interp1d(x_old, moments_array[:, i], kind='linear')
            resampled[:, i] = f(x_new)
        moments_array = resampled

    return imu_array, moments_array


def load_maryam_dataset(imu_root, moment_root):
    """
    Load all Maryam subjects.
    Returns list of (subject_id, imu_array, moments_array)
    """
    subjects = sorted(os.listdir(imu_root))
    dataset = []

    for subj in subjects:
        if subj.startswith('.'):
            continue
        imu_path = os.path.join(imu_root, subj)
        # Match subject name between IMU and moments folders
        # IMU: May8_01, Moments: May08_01 — try both
        mom_path = os.path.join(moment_root, subj)
        if not os.path.exists(mom_path):
            # Try zero-padded version
            parts = subj.split('_')
            if len(parts) == 2:
                month_day = parts[0]
                num = parts[1]
                # Try padding month number
                for alt in [f"May0{month_day[3:]}_{num}",
                            f"May{month_day[3:].zfill(2)}_{num}"]:
                    alt_path = os.path.join(moment_root, alt)
                    if os.path.exists(alt_path):
                        mom_path = alt_path
                        break

        imu_arr, mom_arr = load_maryam_subject(imu_path, mom_path)
        if imu_arr is not None:
            dataset.append((subj, imu_arr, mom_arr))
            print(f"Loaded {subj}: IMU={imu_arr.shape}, Moments={mom_arr.shape}")

    return dataset


# ─────────────────────────────────────────
# WINDOWING
# ─────────────────────────────────────────
def create_windows(imu_array, moments_array, window_size=WINDOW_SIZE,
                   stride=WINDOW_STRIDE):
    """
    Slide a window over the time series.
    Returns: X (n_windows, window_size, n_features)
             y (n_windows, n_moments)
    """
    X, y = [], []
    n = imu_array.shape[0]

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X.append(imu_array[start:end, :])
        # Use the moment at the center of the window
        mid = start + window_size // 2
        y.append(moments_array[mid, :])

    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# IMAGE ENCODING (for DNN / VGG16)
# ─────────────────────────────────────────
def encode_window_as_image(window, image_size=IMAGE_SIZE):
    """
    Encode a (window_size x n_features) window as an RGB image.
    Following Liew 2021: map time→height, features→width, use 3 channels.
    Returns: (image_size, image_size, 3) float32 array
    """
    from scipy.ndimage import zoom

    # Normalize window to [0, 1]
    w_min = window.min(axis=0, keepdims=True)
    w_max = window.max(axis=0, keepdims=True)
    w_range = np.where((w_max - w_min) == 0, 1, w_max - w_min)
    normalized = (window - w_min) / w_range  # (window_size, n_features)

    # Split features into 3 channels (R, G, B)
    n_feat = normalized.shape[1]
    chunk = n_feat // 3
    R = normalized[:, :chunk]
    G = normalized[:, chunk:2*chunk]
    B = normalized[:, 2*chunk:]

    # Pad B if needed
    if B.shape[1] < chunk:
        B = np.pad(B, ((0, 0), (0, chunk - B.shape[1])))

    # Stack and resize to image_size x image_size
    img = np.stack([R, G, B], axis=-1)  # (window_size, chunk, 3)

    # Zoom to target size
    zoom_h = image_size / img.shape[0]
    zoom_w = image_size / img.shape[1]
    img_resized = zoom(img, (zoom_h, zoom_w, 1), order=1)

    return img_resized.astype(np.float32)


def encode_dataset_as_images(X):
    """Encode all windows as images. X: (n, window_size, n_features)"""
    return np.array([encode_window_as_image(X[i]) for i in range(len(X))])


# ─────────────────────────────────────────
# LOSO SPLIT
# ─────────────────────────────────────────
def loso_split(dataset, test_subject_idx):
    """
    Leave-One-Subject-Out split.
    dataset: list of (subj_id, imu_array, moments_array)
    Returns: X_train, y_train, X_test, y_test
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i, (subj, imu, moments) in enumerate(dataset):
        X_win, y_win = create_windows(imu, moments)
        if i == test_subject_idx:
            X_test.append(X_win)
            y_test.append(y_win)
        else:
            X_train.append(X_win)
            y_train.append(y_win)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    # Normalize using training set statistics
    n_train, ws, nf = X_train.shape
    X_train_2d = X_train.reshape(-1, nf)
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_train = X_train_2d.reshape(n_train, ws, nf)

    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, nf)
    X_test_2d = scaler.transform(X_test_2d)
    X_test = X_test_2d.reshape(n_test, ws, nf)

    return X_train, y_train, X_test, y_test, scaler


# ─────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────
if __name__ == '__main__':
    # Test with Maryam dataset paths on Rorqual
    IMU_ROOT = '/scratch/yasir071/maryam_data/Maryam_Dataset/IMU_100Hz_AllSubjects'
    MOM_ROOT = '/scratch/yasir071/maryam_data/Maryam_Dataset/JointMoments'

    print("Loading Maryam dataset...")
    dataset = load_maryam_dataset(IMU_ROOT, MOM_ROOT)
    print(f"\nTotal subjects loaded: {len(dataset)}")

    print("\nTesting LOSO split for subject 0...")
    X_tr, y_tr, X_te, y_te, scaler = loso_split(dataset, test_subject_idx=0)
    print(f"X_train: {X_tr.shape}, y_train: {y_tr.shape}")
    print(f"X_test:  {X_te.shape}, y_test:  {y_te.shape}")

    print("\nTesting image encoding...")
    X_img = encode_dataset_as_images(X_tr[:10])
    print(f"Image batch shape: {X_img.shape}")
    print("\ndata_prep.py OK!")