# functions related to gesture space and gesture type identification
import pandas as pd
import numpy as np
from scipy import stats, signal
import math
from pandas import Series


# defines gesture space as per
# https://www.researchgate.net/figure/Gesture-space-according-to-McNeill-1992-p-89_fig2_355412247
# definitions are based on detection of pose landmarks, world coordinates (in cm) as well as based on pixels

# this will assign landmarks to spaces as defined below
# the assignment uses wrist location - seems more stable in terms of tracking
# finger-based assignment is an alternative
# it seems like fingers, e.g. tip of the index , would cross over spaces more casually - not be desirable.
# this assignment is to be checked against presence of movement later in determining gesture type

def g_space_assign(df):
    columnss = ['LEFT_HAND_SPACE', 'LEFT_HAND_PIX', 'LEFT_HAND_CM',
                'RIGHT_HAND_SPACE', 'RIGHT_HAND_PIX', 'RIGHT_HAND_CM',
                'C_CENTER', 'CENTER', 'PERIPHERY']

    # to write hand location categories
    l_hand = []
    r_hand = []
    l_hand_px = []
    l_hand_cm = []
    r_hand_px = []
    r_hand_cm = []
    # for drawing
    c_cent = []
    center = []
    periphery = []

    for idx in range(len(df)):

        if pd.isna(df.loc[idx, ['NECK_Y', 'MHIP_Y', 'X_LEFT_EAR', 'X_RIGHT_EAR', 'X_RIGHT_HIP', 'X_LEFT_HIP',
                                'X_RIGHT_SHOULDER', 'X_LEFT_SHOULDER', 'Y_LEFT_EYE',
                                'Y_NOSE', 'Y_RIGHT_WRIST', 'Y_LEFT_WRIST']]).any() == False:
            # main measurements
            bodycenter = df.iloc[idx]['NECK_Y'] - (df.iloc[idx]['NECK_Y'] - df.iloc[idx]['MHIP_Y']) / 2
            facewidth = df.iloc[idx]['X_LEFT_EAR'] - df.iloc[idx]['X_RIGHT_EAR']

            # derive spaces from main measurements
            # center-center
            c_cent_xmin = df.iloc[idx]['X_RIGHT_SHOULDER']
            c_cent_xmax = df.iloc[idx]['X_LEFT_SHOULDER']
            c_cent_length = c_cent_xmax - c_cent_xmin
            c_cent_ymin = bodycenter - c_cent_length / 2
            c_cent_ymax = bodycenter + c_cent_length / 2

            # center
            cent_xmin = df.iloc[idx]['X_RIGHT_SHOULDER'] - abs(
                df.iloc[idx]['X_RIGHT_HIP'] - df.iloc[idx]['X_LEFT_HIP']) / 2
            cent_xmax = df.iloc[idx]['X_LEFT_SHOULDER'] + abs(
                df.iloc[idx]['X_RIGHT_HIP'] - df.iloc[idx]['X_LEFT_HIP']) / 2
            cent_ymin = df.iloc[idx]['Y_LEFT_SHOULDER']
            cent_ymax = df.iloc[idx]['Y_RIGHT_HIP']

            # periphery
            peri_ymin = df.iloc[idx]['Y_LEFT_EYE']
            peri_ymax = cent_ymax + abs(peri_ymin - cent_ymin) / 2
            peri_xmax = cent_xmax + facewidth
            peri_xmin = cent_xmin - facewidth

            # anything else can be classified as extreme periphery
            # below also classifies wrist location from mid-hip - arbitrary distance cutoff
            if abs(df.iloc[idx]['Z_W_LEFT_WRIST']) >= 32:
                l_wr_in_cm = 'Away'

            else:
                l_wr_in_cm = 'Close'

            if abs(df.iloc[idx]['Z_W_RIGHT_WRIST']) >= 32:
                r_wr_in_cm = 'Away'

            else:
                r_wr_in_cm = 'Close'

            # below does the same but using pixels - arbitrary pixel cutoff
            if df.iloc[idx]['Y_RIGHT_WRIST'] >= 865:
                r_wr_in_px = 'Down'  # meaning the hand reaches towards the other participant on the table

            elif 670 <= df.iloc[idx]['Y_RIGHT_WRIST'] < 865:
                r_wr_in_px = 'In place'

            elif 480 <= df.iloc[idx]['Y_RIGHT_WRIST'] < 670:
                r_wr_in_px = 'Up'

            else:
                r_wr_in_px = 'Extra Up'

            if df.iloc[idx]['Y_LEFT_WRIST'] >= 865:
                l_wr_in_px = 'Down'

            elif 670 <= df.iloc[idx]['Y_LEFT_WRIST'] < 865:
                l_wr_in_px = 'In place'

            elif 480 <= df.iloc[idx]['Y_LEFT_WRIST'] < 670:
                l_wr_in_px = 'Up'

            else:
                l_wr_in_px = 'Extra Up'

            l_hand_px.append(l_wr_in_px)
            l_hand_cm.append(l_wr_in_cm)
            r_hand_px.append(r_wr_in_px)
            r_hand_cm.append(r_wr_in_cm)

            if c_cent_xmin < df.iloc[idx]['X_RIGHT_WRIST'] < c_cent_xmax and \
                    c_cent_ymin < df.iloc[idx]['Y_RIGHT_WRIST'] < c_cent_ymax:

                r_hand.append('Center-Center')

            elif cent_xmin < df.iloc[idx]['X_RIGHT_WRIST'] < cent_xmax and \
                    cent_ymin < df.iloc[idx]['Y_RIGHT_WRIST'] < cent_ymax:

                r_hand.append('Center')

            elif peri_xmin < df.iloc[idx]['X_RIGHT_WRIST'] < peri_xmax and \
                    peri_ymin < df.iloc[idx]['Y_RIGHT_WRIST'] < peri_ymax:

                r_hand.append('Periphery')

            else:
                r_hand.append('Ex-Periphery')  # extreme periphery

            if c_cent_xmin < df.iloc[idx]['X_LEFT_WRIST'] < c_cent_xmax and \
                    c_cent_ymin < df.iloc[idx]['Y_LEFT_WRIST'] < c_cent_ymax:

                l_hand.append('Center-Center')

            elif cent_xmin < df.iloc[idx]['X_LEFT_WRIST'] < cent_xmax and \
                    cent_ymin < df.iloc[idx]['Y_LEFT_WRIST'] < cent_ymax:

                l_hand.append('Center')

            elif peri_xmin < df.iloc[idx]['X_LEFT_WRIST'] < peri_xmax and \
                    peri_ymin < df.iloc[idx]['Y_LEFT_WRIST'] < peri_ymax:

                l_hand.append('Periphery')

            else:
                l_hand.append('Ex-Periphery')

        else:
            c_cent_xmin, c_cent_xmax, c_cent_ymin, c_cent_ymax, cent_xmin, cent_xmax, cent_ymin, cent_ymax, peri_xmin, \
            peri_xmax, peri_ymin, peri_ymax = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
                                              np.nan, np.nan, np.nan, np.nan, np.nan

            l_hand_px.append(np.nan)
            l_hand_cm.append(np.nan)
            r_hand_px.append(np.nan)
            r_hand_cm.append(np.nan)
            l_hand.append(np.nan)
            r_hand.append(np.nan)

        c_cent.append(np.array([int(c_cent_xmin), int(c_cent_xmax), int(c_cent_ymin), int(c_cent_ymax)]))
        center.append(np.array([int(cent_xmin), int(cent_xmax), int(cent_ymin), int(cent_ymax)]))
        periphery.append(np.array([int(peri_xmin), int(peri_xmax), int(peri_ymin), int(peri_ymax)]))

        g_space = pd.DataFrame()
        g_space[0] = l_hand
        g_space[1] = l_hand_px
        g_space[2] = l_hand_cm
        g_space[3] = r_hand
        g_space[4] = r_hand_px
        g_space[5] = r_hand_cm
        g_space[6] = c_cent
        g_space[7] = center
        g_space[8] = periphery
        g_space.columns = columnss

    return g_space


# to determine hand shape to estimate deixis based on the distance between the tip of the index and the ring finger for both hands
def deic_shape(df, dist_ts):
    columnssss = ["LEFT_DEIC_DIS", "LEFT_DEIC_DET", "RIGHT_DEIC_DIS", "RIGHT_DEIC_DET"]

    left_deic_dis = []
    left_deic_det = []
    right_deic_dis = []
    right_deic_det = []

    for j in range(len(df)):

        # left # ordered list of columns to call with col index later <= reminder
        if pd.isna(df.loc[j, ['X_INDEX_FINGER_TIP_L', 'Y_INDEX_FINGER_TIP_L', 'Z_INDEX_FINGER_TIP_L',
                              'X_RING_FINGER_TIP_L', 'Y_RING_FINGER_TIP_L', 'Z_RING_FINGER_TIP_L']]).all() == False:

            euc_norm = math.sqrt((df.iloc[j, 0] - df.iloc[j, 3]) ** 2 +
                                 (df.iloc[j, 1] - df.iloc[j, 4]) ** 2 +
                                 (df.iloc[j, 2] - df.iloc[j, 5]) ** 2)

            left_deic_dis.append(int(euc_norm))

            if euc_norm >= dist_ts:

                left_deic_det.append("Pointy")

            else:

                left_deic_det.append(np.nan)

        else:

            left_deic_dis.append(np.nan)
            left_deic_det.append(np.nan)

        # right
        if pd.isna(df.loc[j, ['X_INDEX_FINGER_TIP_R', 'Y_INDEX_FINGER_TIP_R', 'Z_INDEX_FINGER_TIP_R',
                              'X_RING_FINGER_TIP_R', 'Y_RING_FINGER_TIP_R', 'Z_RING_FINGER_TIP_R']]).all() == False:

            euc_norm2 = math.sqrt((df.iloc[j, 6] - df.iloc[j, 9]) ** 2 +
                                  (df.iloc[j, 7] - df.iloc[j, 10]) ** 2 +
                                  (df.iloc[j, 8] - df.iloc[j, 11]) ** 2)

            right_deic_dis.append(int(euc_norm2))

            if euc_norm2 >= dist_ts:

                right_deic_det.append("Pointy")

            else:

                right_deic_det.append(np.nan)

        else:

            right_deic_dis.append(np.nan)
            right_deic_det.append(np.nan)

    df2 = pd.DataFrame()
    df2[0] = left_deic_dis
    df2[1] = left_deic_det
    df2[2] = right_deic_dis
    df2[3] = right_deic_det

    df2.columns = columnssss

    return df2


# this determines gesture type based on hand shape, position, duration, peak related info
# first representational vs. non-representational distinction
# then within-representational distinctions

def g_type(df, beat_dur, p_no, p_prom, p_width, *, hand):
    nn = 1  # counter
    df[13] = np.nan

    if Series.last_valid_index(df.iloc[:, 0]):

        while nn <= int(df.iloc[Series.last_valid_index(df.iloc[:, 0]), 0]):

            idx_list = np.array(df.index[df.iloc[:, 0] == nn])  # indexes

            b = idx_list[0]
            e = idx_list[-1] + 1

            # interim gesture type decisions based on parameters contained in the following

            level_1 = []
            level_2 = []

            # duration

            if len(df.iloc[b:e, 0]) <= beat_dur:

                level_1.append("Beat")

            else:

                level_1.append("Rep")

            # peak number

            if df.iloc[b:e, 1].count() <= p_no:

                level_1.append("Beat")

            else:

                level_1.append("Rep")

            # peak prominence

            if (df.iloc[b:e, 1].dropna() >= p_prom).any() == True:

                level_1.append("Rep")

            else:

                level_1.append("Beat")

            # peak width

            if (df.iloc[b:e, 2].dropna() >= p_width).any() == True:

                level_1.append("Rep")

            else:

                level_1.append("Beat")

            if stats.mode(level_1)[0][0] == "Beat":

                interim = "Beat"

            else:

                interim = "Rep"

            if interim == "Beat":

                if hand == "left":

                    # pixel count (only y dimension)

                    if stats.mode(df.iloc[b:e, 4].dropna())[0][0] == "Up" or \
                            stats.mode(df.iloc[b:e, 4].dropna())[0][0] == "Extra Up":
                        interim = "Rep"

                if hand == "right":

                    if stats.mode(df.iloc[b:e, 7].dropna())[0][0] == "Up" or \
                            stats.mode(df.iloc[b:e, 7].dropna())[0][0] == "Extra Up":
                        interim = "Rep"

                if interim == "Beat":

                    df.iloc[b:e, 13] = interim

                else:

                    pass

            # Representational vs. non-representational gesture decision is made above
            # The following tries to distinguish deictics from iconics/metaphorics

            if interim == "Rep":

                if hand == "left":

                    # pixel count (only y dimension)

                    if stats.mode(df.iloc[b:e, 4].dropna())[0][0] == "Up" or \
                            stats.mode(df.iloc[b:e, 4].dropna())[0][0] == "Extra Up" or \
                            stats.mode(df.iloc[b:e, 4].dropna())[0][0] == "Down":

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                    # distance from mid hip

                    if stats.mode(df.iloc[b:e, 5].dropna())[0][0] == "Away":

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                    # hand space

                    if stats.mode(df.iloc[b:e, 3].dropna())[0][0] == "Periphery" or \
                            stats.mode(df.iloc[b:e, 3].dropna())[0][0] == "Ex-Periphery":

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                    # hand shape

                    if df.iloc[b:e, 10].count() >= (len(df.iloc[b:e, 0])) / 2:

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                if hand == "right":

                    # pixel count (only y dimension)

                    if stats.mode(df.iloc[b:e, 7].dropna())[0][0] == "Up" or \
                            stats.mode(df.iloc[b:e, 7].dropna())[0][0] == "Extra Up" or \
                            stats.mode(df.iloc[b:e, 7].dropna())[0][0] == "Down":

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                        # distance from mid hip

                    if stats.mode(df.iloc[b:e, 8] == "Away"):

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                    # hand space

                    if stats.mode(df.iloc[b:e, 6].dropna())[0][0] == "Periphery" or \
                            stats.mode(df.iloc[b:e, 6].dropna())[0][0] == "Ex-Periphery":

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                    # hand shape

                    if df.iloc[b:e, 12].count() >= (len(df.iloc[b:e, 0])) / 2:

                        level_2.append("Deictic")

                    else:

                        level_2.append("Icon/Meta")

                final = stats.mode(level_2)[0][0]

                df.iloc[b:e, 13] = final

            nn += 1

    if hand == 'left':

        df.rename(columns={13: 'G_TYPE_LEFT'}, inplace=True)

    elif hand == 'right':

        df.rename(columns={13: 'G_TYPE_RIGHT'}, inplace=True)

    return df
