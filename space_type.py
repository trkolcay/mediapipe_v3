# functions related to gesture space and gesture type identification

# defines gesture space as per
# https://www.researchgate.net/figure/Gesture-space-according-to-McNeill-1992-p-89_fig2_355412247
# definitions are based on detection of pose landmarks and based on pixels
# to be run inside another loop that assigns spaces - this loop won't append a list

def gesture_space_define(idx, df):
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
        cent_xmin = df.iloc[idx]['X_RIGHT_SHOULDER'] - (df.iloc[idx]['X_RIGHT_HIP'] - df.iloc[idx]['X_RIGHT_SHOULDER'])
        cent_xmax = df.iloc[idx]['X_LEFT_SHOULDER'] + (df.iloc[idx]['X_LEFT_SHOULDER'] - df.iloc[idx]['X_LEFT_HIP'])
        cent_ymin = df.iloc[idx]['Y_RIGHT_HIP']
        cent_ymax = df.iloc[idx]['Y_LEFT_SHOULDER']

        # periphery
        peri_ymax = df.iloc[idx]['Y_LEFT_EYE'] + (df.iloc[idx]['Y_LEFT_EYE'] - df.iloc[idx]['Y_NOSE'])
        peri_ymin = cent_ymin - (peri_ymax - cent_ymax)
        peri_xmax = cent_xmax + facewidth
        peri_xmin = cent_xmin - facewidth

        # anything else can be classified as extreme periphery
        # below also classifies wrist location from mid-hip - arbitrary distance cutoff
        if df.iloc[idx, 'Z_W_LEFT_WRIST'] >= 25:
            l_wr_in_cm = 'Away'

        else:
            l_wr_in_cm = 'Close'

        if df.iloc[idx, 'Z_W_RIGHT_WRIST'] >= 25:
            r_wr_in_cm = 'Away'

        else:
            r_wr_in_cm = 'Close'

        # below does the same but using pixels - arbitrary pixel cutoff
        if df.iloc[idx, 'Y_RIGHT_WRIST'] >= 920:
            r_wr_in_px = 'Down' # meaning the hand reaches towards the other participant on the table

        elif 750 <= df.iloc[idx, 'Y_RIGHT_WRIST'] < 920:
            r_wr_in_px = 'In place'

        elif 480 <= df.iloc[idx, 'Y_RIGHT_WRIST'] < 750:
            r_wr_in_px = 'Up'

        else:
            r_wr_in_px = 'Extra Up'

        if df.iloc[idx, 'Y_LEFT_WRIST'] >= 910:
            l_wr_in_px = 'Down'

        elif 740 <= df.iloc[idx, 'Y_LEFT_WRIST'] < 910:
            l_wr_in_px = 'In place'

        elif 470 <= df.iloc[idx, 'Y_LEFT_WRIST'] < 740:
            l_wr_in_px = 'Up'

        else:
            l_wr_in_px = 'Extra Up'

    else:
        c_cent_xmin, c_cent_xmax, c_cent_ymin, c_cent_ymax, cent_xmin, cent_xmax, cent_ymin, cent_ymax, \
        peri_xmin, peri_xmax, peri_ymin, peri_ymax, r_wr_in_px, r_wr_in_cm, l_wr_in_px, l_wr_in_cm =\
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
        np.nan, np.nan, np.nan, np.nan


    return c_cent_xmin, c_cent_xmax, c_cent_ymin, c_cent_ymax, cent_xmin, cent_xmax, \
           cent_ymin, cent_ymax, peri_xmin, peri_xmax, peri_ymin, peri_ymax, r_wr_in_px, \
           r_wr_in_cm, l_wr_in_px, l_wr_in_cm

# this will assign landmarks to spaces as defined above
# the assignment uses wrist location - seems more stable in terms of tracking
# finger-based assignment is an alternative
# it feels like fingers, e.g. tip of the index , would cross over spaces more casually. This may not be desirable.
# this assignment is to be checked against presence of movement later in determining gesture type
def gesture_space_assign(df):
    columnss = ['LEFT_HAND_SPACE', 'LEFT_HAND_PIX', 'LEFT_HAND_CM',
                'RIGHT_HAND_SPACE', 'RIGHT_HAND_PIX', 'RIGHT_HAND_CM']
    l_hand = []
    r_hand = []
    l_hand_px = []
    l_hand_cm = []
    r_hand_px = []
    r_hand_cm = []
    for idx in range(len(df)):

        if pd.isna(df.loc[idx, ['Y_RIGHT_WRIST', 'Y_LEFT_WRIST', 'X_RIGHT_WRIST', 'X_LEFT_WRIST']].any() == False:

            c_cent_xmin, c_cent_xmax, c_cent_ymin, c_cent_ymax, cent_xmin, cent_xmax, cent_ymin, cent_ymax, \
            peri_xmin, peri_xmax, peri_ymin, peri_ymax, l_wr_in_cm, \
            l_wr_in_px, r_wr_in_cm, r_wr_in_px = gesture_space_define(idx, df)

            l_hand_px.append(l_wr_in_px)
            l_hand_cm.append(l_wr_in_cm)
            r_hand_px.append(r_wr_in_px)
            r_hand_cm.append(r_wr_in_cm)

            if c_cent_xmin < df.iloc[idx, 'X_RIGHT_WRIST'] < c_cent_xmax and
                    c_cent_ymin < df.iloc[idx, 'Y_RIGHT_WRIST'] < c_cent_ymax:

                r_hand.append('Center-Center')

            elif cent_xmin < df.iloc[idx, 'X_RIGHT_WRIST'] < cent_xmax and
                    cent_ymin < df.iloc[idx, 'Y_RIGHT_WRIST'] < cent_ymax:

                r_hand.append('Center')

            elif peri_xmin < df.iloc[idx, 'X_RIGHT_WRIST'] < peri_xmax and
                    peri_ymin < df.iloc[idx, 'Y_RIGHT_WRIST'] < peri_ymax:

                r_hand.append('Periphery')

            else:
                r_hand.append('Ex-Periphery') # extreme periphery

            if c_cent_xmin < df.iloc[idx, 'X_LEFT_WRIST'] < c_cent_xmax and
                    c_cent_ymin < df.iloc[idx, 'Y_LEFT_WRIST'] < c_cent_ymax:

                l_hand.append('Center-Center')

            elif cent_xmin < df.iloc[idx, 'X_LEFT_WRIST'] < cent_xmax and
                    cent_ymin < df.iloc[idx, 'Y_LEFT_WRIST'] < cent_ymax:

                l_hand.append('Center')

            elif peri_xmin < df.iloc[idx, 'X_LEFT_WRIST'] < peri_xmax and
                    peri_ymin < df.iloc[idx, 'Y_LEFT_WRIST'] < peri_ymax:

                l_hand.append('Periphery')

            else:
                l_hand.append('Ex-Periphery')

        else:
            l_hand.append(np.nan)
            r_hand.append(np.nan)

    g_space = pd.DataFrame()
    g_space[0] = l_hand
    g_space[1] = l_hand_px
    g_space[2] = l_hand_cm
    g_space[3] = r_hand
    g_space[4] = r_hand_px
    g_space[5] = r_hand_cm
    g_space.columns = columnss
    return g_space





























































# to answer which space was mostly used
def space_used(df):
    mainspace = []
    for sp in df:
        if sp > 40:
            mainspace.append(4)

        elif sp > 30:
            mainspace.append(3)

        else:
            mainspace.append(sp)

    space_use = stats.mode(mainspace)
    return space_use
