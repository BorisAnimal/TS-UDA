from glob2 import glob
import os
from os.path import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

feature_columns = """acc_x acc_y acc_z gyr_x gyr_y gyr_z mag_x mag_y mag_z orient_w orient_x orient_y orient_z grav_x 
grav_y grav_z lin_acc_x lin_acc_y lin_acc_z pressure ignore ignore""".split()

label_columns = ["Coarse", "Fine", "Road", "Traffic", "Tunnels", "Social", "Food"]

coarse_label_mapping = "Null Still Walking Run Bike Car Bus Train Subway".split()

label_column = "Coarse"


def __get_windows(x, window_size=500, step=250):
    # dataset (np.array), window_size, step, data_type (original)
    tSamples = []
    for i in range(window_size, len(x), step):
        #             # Due to step < window_size (in common case)
        #             if len(tSamples) - i < window_size:
        #                 break
        curRange = x[i - window_size: i]
        tSamples.append(curRange)
    #             # Reshape test:
    #             a = np.array([[1,2,3,0], [4,5,6,0], [7,8,9,0]])
    #             print(a.shape)
    #             a.reshape((3,2,2))
    #             a = array([[[1, 2],[3, 0]],
    #                        [[4, 5],[6, 0]],
    #                        [[7, 8],[9, 0]]])
    return tSamples


def windows_split(x, y, window_size=500, step=250, flatten=False):
    """
    :param x: np.array
    :param y: np.array
    :param flatten: if true, returns 2dim np.array, else 3dim (num_samples, window_size, num_columns)
    :return: np.arrays!!
    """

    count, prev_val, prev_ind = 0, 0, 0
    slices = []
    assert len(x) == len(y)
    #     Collect indices of slices
    for ind, i in enumerate(y.flatten()):
        # will be appended only long enough slices
        if prev_val != i:
            if count >= window_size:
                slices.append((prev_ind, ind, prev_val))  # (slice_start, slice_end, label)
            prev_val = i
            count = 0
            prev_ind = ind
        count += 1
    # Add last:
    slices.append((prev_ind, ind, prev_val))
    slices = sorted(slices, key=lambda k: k[1] - k[0])

    xw = []
    yw = []
    for (slice_start, slice_end, label) in slices:
        for window in __get_windows(x[slice_start:slice_end], window_size=window_size, step=step):
            if flatten:
                xw.append(window.flatten())
            else:
                xw.append(window)
            yw.append(label)
    xw = np.array(xw)
    yw = np.array(yw).reshape(-1, 1)
    return xw, yw


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def smooth(x, w):
    """
    Smooth by last dimension.
    :param x: numpy array, data
    :param w: int, window len
    :return: numpy array
    """
    if len(x.shape) == 1:
        return moving_average(x, w)
    else:
        return np.array([smooth(row, w) for row in x])


if __name__ == '__main__':
    # 1. read folders "220617", "260617", ...
    # loop = tqdm(glob('./data/raw/*'))
    # loop = tqdm(glob('F:/datasets/shl/shl-3users/*'))
    loop = tqdm(list(glob('F:/datasets/shl_user1_hips/user1/*')))
    for d in loop:
        y_file = "Label.txt"
        # modes = ["Hand_Motion.txt", "Torso_Motion.txt"]
        modes = ["Hips_Motion.txt"]
        for x_file in modes:
            #     Load X, Y
            loop.set_description("Loading")
            X_path, Y_path = join(d, x_file), join(d, y_file)
            # X = pd.DataFrame(np.loadtxt(X_path), columns=["Time(ms)"] + feature_columns)
            X = pd.read_csv(X_path, sep=' ', header=None, names=["Time(ms)"] + feature_columns)
            X.set_index(pd.to_datetime(X['Time(ms)'], unit='ms'), inplace=True)

            # 3. select columns
            X = X.iloc[:, 1:10]  # Only [acc_x acc_y acc_z gyr_x gyr_y gyr_z mag_x mag_y mag_z] columns
            X_col = list(X.columns)
            # Y = pd.DataFrame(np.loadtxt(Y_path), columns=["Time(ms)"] + label_columns)
            Y = pd.read_csv(Y_path, sep=' ', header=None, names=["Time(ms)"] + label_columns)
            Y.set_index(pd.to_datetime(Y['Time(ms)'], unit='ms'), inplace=True)
            #     Y = to_categorical(data['Fine'], num_classes=len(labels))

            assert (X.shape[0] == Y.shape[0])

            Y = pd.DataFrame(Y[label_column])
            data = pd.concat([X, Y], join='inner', axis=1)

            # 7. Set desired frequency 100 -> 50 (Hz)
            # data = data.iloc[::2]

            # 2. drop nan data
            data.dropna(inplace=True)
            X, Y = data[X_col].values, data[[label_column]].values

            # 4. normalize data
            loop.set_description("Amplitudes")
            # 4.5 Amplitudes
            X = X.reshape(-1, 3, 3)
            X = np.linalg.norm(X, axis=2)
            X = X.reshape(-1, 3)

            loop.set_description("Normalization")
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)

            # Rescaling (min-max normalization)
            X = (X - X_min) / (X_max - X_min)
            X = X * 2 - 1

            # 5. split on windows according to Label.txt
            loop.set_description("Splitting")
            Xw, Yw = windows_split(X, Y, window_size=500, step=250)

            # 5.5. Change shape to (len, channels, measures)
            Xw = Xw.transpose(0, 2, 1)

            Xfreq = smooth(Xw, 10)

            Xw = smooth(Xw, 3)

            scalers = [2, 3, 5]
            Xscale = np.concatenate([Xw[:, :, ::s] for s in scalers], axis=2)

            # 6. save windows
            loop.set_description("Saving")
            dst_dir = join('./data/shl-source', d[-6:])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            # print("Writing Xw with shape:", Xw.shape)
            prefix = join(dst_dir, x_file.split(".")[0])
            np.save(prefix + ".npy", Xw)
            np.save(prefix + "_scaled.npy", Xscale)
            np.save(prefix + "_freq.npy", Xfreq)
            # print("Writing Yw with shape:", Yw.shape)
            np.save(join(dst_dir, x_file.split(".")[0] + "_labels.npy"), Yw)
