# functions related to gesture space and gesture type identification
import pandas as pd
import numpy as np
from scipy import stats, signal


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

