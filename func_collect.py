# Useful function collection related to distance, velocity and acceleration
import math
from scipy import stats, signal
import numpy as np
import pandas as pd
from pandas import Series


# Euclidean distance for blinks
def euc_dis_blk(point_1, point_2, frameWidth, frameHeight):
    x, y = int(point_1.x * frameWidth), int(point_1.y * frameHeight)
    x1, y1 = int(point_2.x * frameWidth), int(point_2.y * frameHeight)
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Eye markers' distance ratio
def blink_ratio(image, landmark, frameWidth, frameHeight):
    # RIGHT eye coordinates
    # horizontal
    rh_right = landmark[33]
    #    circle_d(image, rh_right)
    rh_left = landmark[155]
    #    circle_d(image, rh_left)
    # vertical line
    rv_top = landmark[158]
    #    circle_d(image, rv_top)
    rv_bottom = landmark[144]
    #    circle_d(image, rv_bottom)
    # LEFT eye coordinates
    # horizontal
    lh_right = landmark[362]
    #    circle_d(image, lh_right)
    lh_left = landmark[249]
    #    circle_d(image, lh_left)
    # vertical
    lv_top = landmark[387]
    #    circle_d(image, lv_top)
    lv_bottom = landmark[380]
    #    circle_d(image, lv_bottom)
    # calculate euclidean distances between outer landmarks
    rh_dis = euc_dis_blk(rh_right, rh_left, frameWidth, frameHeight)
    rv_dis = euc_dis_blk(rv_top, rv_bottom, frameWidth, frameHeight)

    lv_dis = euc_dis_blk(lv_top, lv_bottom, frameWidth, frameHeight)
    lh_dis = euc_dis_blk(lh_right, lh_left, frameWidth, frameHeight)

    re_ratio = rh_dis / rv_dis
    le_ratio = lh_dis / lv_dis

    ratio = (re_ratio + le_ratio) / 2

    return ratio


# below calculates 3D velocity anc acceleration vectors, computes euclidean norm of sums of both, and smooths them
# then segments the nsu vectors of velocity and acceleration into movement/no movement (vel) and amplitude (acc)
# it requires a dataframe with three columns for x, y, z landmarks and produces a dataframe with 12 columns
# dataframe must be named (df.name) in the main file so that appended lists can be associated with that name
# based on movement segmentation and entered threshold it distinguishes between individual gestures and numbers them
def velo_acc_calc(df, fps, vel_threshold, acc_threshold, count_threshold, height, prominence, width, distance):
    column_names = ["VEL_X" + "_" + df.name, "VEL_Y" + "_" + df.name, "VEL_Z" + "_" + df.name,
                    "VEL_NSUM" + "_" + df.name, "VEL_SMOOTH" + "_" + df.name,
                    "ACC_NSUM" + "_" + df.name, "ACC_SMOOTH" + "_" + df.name,
                    "VEL_THRESHOLD" + "_" + df.name, "ACC_THRESHOLD" + "_" + df.name,
                    "GESTURE_FILTER" + "_" + df.name, "GESTURE_NO" + "_" + df.name,
                    "PEAK_PROM" + "_" + df.name, "PEAK_WIDTH" + "_" + df.name]

    # prepare lists to be appended
    # time aligning with nan since velocity cannot be computed for the first frame (no r-1)
    vel_x = [np.nan]
    vel_y = [np.nan]
    vel_z = [np.nan]
    vel_nsum = [np.nan]
    acc_nsum = [np.nan]
    t = 1 / fps

    # main routine - fps to be passed where the functions imported to
    for r in range(1, len(df)):
        if df.iloc[r,].isnull().any() == False and df.iloc[r - 1,].isnull().any() == False:
            # distance divided by framerate gives velocity
            vlc_x = (int(df.iloc[r, 0]) - int(df.iloc[r - 1, 0])) / t
            vlc_y = (int(df.iloc[r, 1]) - int(df.iloc[r - 1, 1])) / t
            vlc_z = (int(df.iloc[r, 2]) - int(df.iloc[r - 1, 2])) / t

            vel_x.append(vlc_x)
            vel_y.append(vlc_y)
            vel_z.append(vlc_z)
        # nan if no motion is tracked (despite interpolation) from hereon
        else:
            vel_x.append(np.nan)
            vel_y.append(np.nan)
            vel_z.append(np.nan)

    # put together
    df2 = pd.DataFrame()
    df2[0] = vel_x
    df2[1] = vel_y
    df2[2] = vel_z

    # calculating norm of sums for 3 velocity vectors and derive acceleration
    for r2 in range(1, len(df2)):
        if df2.iloc[r2, 0:3].isnull().any() == False and df2.iloc[r2 - 1, 0:3].isnull().any() == False:
            # Pythagorean
            euc_norm = math.sqrt((df2.iloc[r2, 0] - df2.iloc[r2 - 1, 0]) ** 2 +
                                 (df2.iloc[r2, 1] - df2.iloc[r2 - 1, 1]) ** 2 +
                                 (df2.iloc[r2, 2] - df2.iloc[r2 - 1, 2]) ** 2)
            vel_nsum.append(int(euc_norm))

        else:
            vel_nsum.append(np.nan)

        if np.isnan([vel_nsum[r2], vel_nsum[r2 - 1]]).any() == False:

            acc_temp_x = (vel_nsum[r2] - vel_nsum[r2 - 1]) / t
            acc_nsum.append(acc_temp_x)

        else:
            acc_nsum.append(np.nan)

    df2[3] = vel_nsum
    # smoothing with Savitzky-Golay filter
    df2[4] = np.round(signal.savgol_filter(df2[3], 15, 5).astype(float), 0)
    df2[5] = acc_nsum
    df2[6] = np.round(signal.savgol_filter(df2[5], 15, 5).astype(float), 0)

    # below segments velocity and acceleration vectors according to the thresholds set
    for i in range(len(df2)):
        if pd.isna(df2.iloc[i, 4]) == False:
            if abs(df2.iloc[i, 4]) >= vel_threshold:
                df2.loc[i, 7] = "Movement"

            else:
                df2.loc[i, 7] = np.nan
        else:
            df2.loc[i, 7] = np.nan

        if pd.isna(df2.iloc[i, 6]) == False:
            if abs(df2.iloc[i, 6]) >= acc_threshold:
                df2.loc[i, 8] = "High Amp"

            else:
                df2.loc[i, 8] = "Low Amp"
        else:
            df2.loc[i, 8] = np.nan

    # short durations of movements or no movement within larger chunks of the other kind are converted (discrete filter)
    df2[9] = np.nan
    df2.iloc[:, 9] = df2.iloc[:, 7]

    for m in range(len(df2) - 1):

        if pd.isna(df2.iloc[m, 9]) == False and pd.isna(df2.iloc[m + 1, 9]) == True:

            idx_temp = np.where(df2.iloc[m + 1:, 9].notna())[0][0]

            if idx_temp <= count_threshold:
                df2.iloc[m:m + count_threshold + 1, 9] = "Movement"

        elif pd.isna(df2.iloc[m, 9]) == True and pd.isna(df2.iloc[m + 1, 9]) == False:
            idx_temp2 = np.where(df2.iloc[m + 1:, 9].isna())[0][0]

            if idx_temp2 <= count_threshold:
                df2.iloc[m:m + count_threshold + 1, 9] = np.nan

    # below numbers individual movements

    c = int(0)
    g_count = []

    for q in range(len(df2) - 1):

        if df2.iloc[q, 9] == 'Movement':

            g_count.append(c)

        else:

            g_count.append(np.nan)

            if df2.iloc[q + 1, 9] == 'Movement':
                c += 1

    g_count.append(np.nan)
    df2[10] = g_count

    df2[11], df2[12] = np.nan, np.nan

    ii = 1

    while ii <= int(df2.iloc[Series.last_valid_index(df2.iloc[:, 10]), 10]):

        idx_list = np.array(df2.index[df2.iloc[:, 10] == ii])

        peak_idx, peak_dict = signal.find_peaks(df2.iloc[idx_list[0]:idx_list[-1] + 1, 4],
                                                height=height, prominence=prominence, width=width, distance=distance)

        temp1 = {'peak_indices': peak_idx}

        peak_dict.update(temp1)

        for pp in range(len(peak_dict['peak_indices'])):
            df2.iloc[idx_list[0] + peak_dict['peak_indices'][pp], 11] = int(np.round(peak_dict['prominences'][pp], 0))

            df2.iloc[idx_list[0] + peak_dict['peak_indices'][pp], 12] = int(np.round(peak_dict['widths'][pp], 0))

        ii += 1

    # rename columns as per above
    df2.columns = column_names

    return df2
