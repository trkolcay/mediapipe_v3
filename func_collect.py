# Useful function collection related to distance, velocity and acceleration

# Euclidean distance for blinks
def euc_dis_blk(point_1, point_2):
    x, y = int(point_1.x * frameWidth), int(point_1.y * frameHeight)
    x1, y1 = int(point_2.x * frameWidth), int(point_2.y * frameHeight)
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# Eye markers' distance ratio
def blink_ratio(image, landmark):
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
    rh_dis = euc_dis_blk(rh_right, rh_left)
    rv_dis = euc_dis_blk(rv_top, rv_bottom)

    lv_dis = euc_dis_blk(lv_top, lv_bottom)
    lh_dis = euc_dis_blk(lh_right, lh_left)

    re_ratio = rh_dis / rv_dis
    le_ratio = lh_dis / lv_dis

    ratio = (re_ratio + le_ratio) / 2

    return ratio


# below calculates 3D velocity anc acceleration vectors, computes euclidean norm of sums of both, and smooths them
# then segments the nsu vectors of velocity and acceleration into movement/no movement (vel) and amplitude (acc)
# it requires a dataframe with three columns for x, y, z landmarks and produces a dataframe with 12 columns
# dataframe must be named (df.name) in the main file so that appended lists can be associated with that name
def velo_acc_calc(df, vel_threshold, acc_threshold):
    column_names = ["VEL_X" + "_" + df.name, "VEL_Y" + "_" + df.name, "VEL_Z" + "_" + df.name,
                    "ACC_X" + "_" + df.name, "ACC_Y" + "_" + df.name, "ACC_Z" + "_" + df.name,
                    "VEL_NSUM" + "_" + df.name, "VEL_SMOOTH" + "_" + df.name,
                    "ACC_NSUM" + "_" + df.name, "ACC_SMOOTH" + "_" + df.name,
                    "VEL_THRESHOLD" + "_" + df.name, "ACC_THRESHOLD" + "_" + df.name]

    # prepare lists to be appended
    # time aligning with nan since velocity cannot be computed for the first frame (no r-1)
    vel_x = [np.nan]
    vel_y = [np.nan]
    vel_z = [np.nan]
    vel_nsum = [np.nan]
    acc_x = [np.nan]
    acc_y = [np.nan]
    acc_z = [np.nan]
    acc_nsum = [np.nan]
    t = 1 / fps

    # main routine - fps and frameWidth to be passed where the functions imported to
    for r in range(1, len(df)):
        if df.iloc[r,].isnull().any() == False and df.iloc[r - 1,].isnull().any() == False:
            # euc_distance distance divided by framerate gives velocity
            vlc_x = (int(df.iloc[r, 0]) - int(df.iloc[r - 1, 0])) / t
            vlc_y = (int(df.iloc[r, 1]) - int(df.iloc[r - 1, 1])) / t
            vlc_z = (int((df.iloc[r, 2] * frameWidth)) - int((df.iloc[r - 1, 2] * frameWidth))) / t

            vel_x.append(vlc_x)
            vel_y.append(vlc_y)
            vel_z.append(vlc_z)
        # nan if no motion is tracked (despite interpolation) from hereon
        else:
            vel_x.append(np.nan)
            vel_y.append(np.nan)
            vel_z.append(np.nan)

        # velocity difference divided by framerate gives acceleration
        if np.isnan([vel_x[r], vel_x[r - 1]]).any() == False:
            acc_temp_x = (vel_x[r] - vel_x[r - 1]) / t
            acc_x.append(acc_temp_x)

        else:
            acc_x.append(np.nan)

        if np.isnan([vel_y[r], vel_y[r - 1]]).any() == False:
            acc_temp_y = (vel_y[r] - vel_y[r - 1]) / t
            acc_y.append(acc_temp_y)

        else:
            acc_y.append(np.nan)

        if np.isnan([vel_z[r], vel_z[r - 1]]).any() == False:
            acc_temp_z = (vel_z[r] - vel_z[r - 1]) / t

            acc_z.append(acc_temp_z)

        else:
            acc_z.append(np.nan)

    # put together
    df2 = pd.DataFrame()
    df2[0] = vel_x
    df2[1] = vel_y
    df2[2] = vel_z
    df2[3] = acc_x
    df2[4] = acc_y
    df2[5] = acc_z

    # calculating norm of sums for 3 velocity vectors and 3 acceleration vectors
    for r2 in range(1, len(df2)):
        if df2.iloc[r2, 0:3].isnull().any() == False and df2.iloc[r2 - 1, 0:3].isnull().any() == False:
            # Pythagorean
            euc_norm = math.sqrt((df2.iloc[r2, 0] - df2.iloc[r2 - 1, 0]) ** 2 +
                               (df2.iloc[r2, 1] - df2.iloc[r2 - 1, 1]) ** 2 +
                               (df2.iloc[r2, 2] - df2.iloc[r2 - 1, 2]) ** 2)

            vel_nsum.append(int(euc_norm))

        else:
            vel_nsum.append(np.nan)

        if df2.iloc[r2, 4:7].isnull().any() == False and df2.iloc[r2 - 1, 4:7].isnull().any() == False:

            acc_temp_nsum = math.sqrt((df2.iloc[r2, 3] - df2.iloc[r2 - 1, 3]) ** 2 +
                                    (df2.iloc[r2, 4] - df2.iloc[r2 - 1, 4]) ** 2 +
                                    (df2.iloc[r2, 5] - df2.iloc[r2 - 1, 5]) ** 2)

            acc_nsum.append(int(acc_temp_nsum))

        else:
            acc_nsum.append(np.nan)

    df2[6] = vel_nsum
    # smoothing with Savitzky-Golay filter
    df2[7] = np.round(signal.savgol_filter(df2[6], 15, 5).astype(float), 0)
    df2[8] = acc_nsum
    df2[9] = np.round(signal.savgol_filter(df2[8], 15, 5).astype(float), 0)

    # below segments velocity and acceleration vectors according to the thresholds set
    for i in range(len(df2)):

        if pd.isna(df2.iloc[i, 7]) == False:

            if df2.iloc[i, 7] >= vel_threshold:

                df2.loc[i, 10] = "Movement"

            else:
                df2.loc[i, 10] = "No Movement"

        else:
            df2.loc[i, 10] = np.nan

        if pd.isna(df2.iloc[i, 9]) == False:

            if df2.iloc[i, 9] >= acc_threshold:

                df2.loc[i, 11] = "High Amp"

            else:
                df2.loc[i, 11] = "Low Amp"

        else:

            df2.loc[i, 11] = np.nan

    # rename columns as per above
    df2.columns = column_names

    return df2