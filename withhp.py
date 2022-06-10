import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import time
import math
from scipy import stats, signal
from os import listdir
from os.path import isfile, join
from func_collect import euc_dis_blk, blink_ratio, velo_acc_calc

# prepare folders for looping
videoinput = "./video_input/shoulder/"
eachfile = [f for f in listdir(videoinput) if isfile(join(videoinput, f))]
csvoutput = "./csv_output/shoulder/"
videooutput = "./video_output/shoulder/"

# mediapipe solutions to be used
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Create columns for mediapipe output
pose_markers = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE',
                'RIGHT_EYE_OUTER',
                'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
                'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
                'RIGHT_ANKLE',
                'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

hand_left_markers = ['WRIST_L', 'THUMB_CPC_L', 'THUMB_MCP_L', 'THUMB_IP_L', 'THUMB_TIP_L', 'INDEX_FINGER_MCP_L',
                     'INDEX_FINGER_PIP_L', 'INDEX_FINGER_DIP_L', 'INDEX_FINGER_TIP_L', 'MIDDLE_FINGER_MCP_L',
                     'MIDDLE_FINGER_PIP_L', 'MIDDLE_FINGER_DIP_L', 'MIDDLE_FINGER_TIP_L', 'RING_FINGER_PIP_L',
                     'RING_FINGER_DIP_L', 'RING_FINGER_TIP_L', 'RING_FINGER_MCP_L', 'PINKY_MCP_L', 'PINKY_PIP_L',
                     'PINKY_DIP_L', 'PINKY_TIP_L']

hand_right_markers = ['WRIST_R', 'THUMB_CPC_R', 'THUMB_MCP_R', 'THUMB_IP_R', 'THUMB_TIP_R', 'INDEX_FINGER_MCP_R',
                      'INDEX_FINGER_PIP_R', 'INDEX_FINGER_DIP_R', 'INDEX_FINGER_TIP_R', 'MIDDLE_FINGER_MCP_R',
                      'MIDDLE_FINGER_PIP_R', 'MIDDLE_FINGER_DIP_R', 'MIDDLE_FINGER_TIP_R', 'RING_FINGER_PIP_R',
                      'RING_FINGER_DIP_R', 'RING_FINGER_TIP_R', 'RING_FINGER_MCP_R', 'PINKY_MCP_R', 'PINKY_PIP_R',
                      'PINKY_DIP_R', 'PINKY_TIP_R']

# for estimations - not a mediapipe solution unlike above
lean_markers = ["LEAN_X", "LEAN_Y", "LEAN_Z", "TORSO_LEAN", "ALTER_LEAN", "NECK_X", "NECK_Y", "NECK_Z",
                "MHIP_X", "MHIP_Y", "MHIP_Z"]
head_markers = ["HEAD_X", "HEAD_Y", "HEAD_Z", "HEAD_DIRECTION", "HEAD_TILT", "BLINK_RATIO", "BLINK", "FOREHEAD_X",
                "FOREHEAD_Y", "FOREHEAD_Z"]
time_row = ['TIME']

pose_markers2 = []
for mark in pose_markers:
    for pos in ['X', 'Y', 'Z', 'Vis']:
        nm = pos + "_" + mark
        pose_markers2.append(nm)

w_pose_markers = ['X_W_LEFT_WRIST', 'Y_W_LEFT_WRIST', 'Z_W_LEFT_WRIST', ' Vis_W_LEFT_WRIST',
                  'X_W_RIGHT_WRIST', 'Y_W_RIGHT_WRIST', 'Z_W_RIGHT_WRIST', 'Vis_W_RIGHT_WRIST',
                  'X_W_LEFT_INDEX', 'Y_W_LEFT_INDEX', 'Z_W_LEFT_INDEX', ' Vis_W_LEFT_INDEX',
                  'X_W_RIGHT_INDEX', 'Y_W_RIGHT_INDEX', 'Z_W_RIGHT_INDEX', 'Vis_W_RIGHT_INDEX']

twohands_markers = hand_left_markers + hand_right_markers
twohands_markers2 = []
for mark in twohands_markers:
    for pos in ['X', 'Y', 'Z']:
        nm2 = pos + "_" + mark
        twohands_markers2.append(nm2)

markerz = pose_markers2 + lean_markers + twohands_markers2 + head_markers + w_pose_markers

# uncomment and adapt if interested in writing all 468 face landmarks (+10 if refine_landmarks is 1)
# face_markers = []
# for val in range(1, 468):
#    face_markers += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

whole_row = time_row + markerz  # + face_markers

# loop through all the video files
for ff in eachfile:
    timee = 0  # this will contain time information in the loop
    start_time = time.time()  # real time
    print(f'###########{ff} started at {time.strftime("%H:%M:%S", time.localtime())}###########')

    # initialise a csv
    with open(csvoutput + ff[:-4] + '.csv', mode='w', newline='') as cs:
        csv_writer = csv.writer(cs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(whole_row)

    # Read video frames
    cap = cv2.VideoCapture(videoinput + ff)

    # extract video properties
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # below deals with camera calibration. I am a little out of my depth here...
    # they are used in head, torso estimation using various matrices
    # camera matrix
    focal_length = 1 * frameWidth
    cam_matrix = np.array([[focal_length, 0, frameHeight / 2],
                           [0, focal_length, frameWidth / 2],
                           [0, 0, 1]])

    # distortion parameter = no distortion
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # prepare video output with drawings
    samplerate = fps  # make it the same as current video # may need to reduce quality for faster processing later
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # (*'mp4v') also an option
    vidout = cv2.VideoWriter(videooutput + ff[:-4] + '.avi', fourcc, fps=samplerate,
                             frameSize=(int(frameWidth), int(frameHeight)))

    # main routine
    with mp_holistic.Holistic(static_image_mode=False,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.6,
                              model_complexity=2,
                              enable_segmentation=False,
                              smooth_landmarks=True,
                              refine_face_landmarks=True) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            start = time.time()  # curious how long the processing will take on cpu
            if not success:
                print("No frame could be read.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # mediapipe requirement
            image.flags.writeable = False  # apparently improves performance
            results = holistic.process(image)  # main container of data
            # print(results.pose_landmarks)

            # Draw landmark annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # drawing face
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            # landmark_drawing_spec=None,
            # connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # drawing pose
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
            # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # drawing hands
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

            # save landmarks in the right order to write into the csv as initialised above
            # reverting normalised values and rounding visibility
            pose_row = []
            lean_row = []
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array([[int(landmark.x * frameWidth),
                               int(landmark.y * frameHeight),
                               int(landmark.z * frameWidth),
                               np.round(landmark.visibility, 6)] for landmark in pose]).flatten())

                # for estimating torso leans from shoulders and hips
                # hips are problematic here as mp predicts them without really detecting them, which introduces jitters
                # the reason I am passing them is that solvePnP requires more than two points. I had intended to use
                # only left and right shoulders here (11 and 12)...
                lean_2d = []
                lean_3d = []
                neck_3d = []
                midhip_3d = []
                for idx, lm in enumerate(pose):
                    l_shoulder_2d = ()
                    l_shoulder_3d = ()
                    r_shoulder_2d = ()
                    r_shoulder_3d = ()
                    l_hip_2d = ()
                    l_hip_3d = ()
                    r_hip_2d = ()
                    r_hip_3d = ()
                    if idx == 11 or idx == 12 or idx == 23 or idx == 24:  # landmark numbers
                        if idx == 11:
                            l_shoulder_2d = (lm.x * frameWidth, lm.y * frameHeight)  # for later use in projection
                            l_shoulder_3d = (lm.x * frameWidth, lm.y * frameHeight, lm.z * frameWidth)  # random 2000
                            l_shoulder_dis = lm.z * frameWidth
                        if idx == 12:
                            r_shoulder_2d = (lm.x * frameWidth, lm.y * frameHeight)
                            r_shoulder_3d = (lm.x * frameWidth, lm.y * frameHeight, lm.z * frameWidth)
                            r_shoulder_dis = lm.z * frameWidth

                        if len(l_shoulder_3d) and len(r_shoulder_3d) != 0:
                            # locate neck half way between left and right shoulders
                            neck_x = r_shoulder_3d[0] + ((l_shoulder_3d[0] - r_shoulder_3d[0]) / 2)
                            neck_y = r_shoulder_3d[1] + ((l_shoulder_3d[1] - r_shoulder_3d[1]) / 2)
                            neck_z = l_shoulder_3d[2] + ((r_shoulder_3d[2] - l_shoulder_3d[2]) / 2)
                            neck_3d.append([neck_x, neck_y, neck_z])

                        else:
                            neck_3d.append([np.nan] * 3)

                        if idx == 23:
                            l_hip_2d = (lm.x * frameWidth, lm.y * frameHeight)  # for later use in projection
                            l_hip_3d = (lm.x * frameWidth, lm.y * frameHeight, lm.z * frameWidth)  # random 2000

                        if idx == 24:
                            r_hip_2d = (lm.x * frameWidth, lm.y * frameHeight)
                            r_hip_3d = (lm.x * frameWidth, lm.y * frameHeight, lm.z * frameWidth)

                        if len(l_hip_3d) and len(r_hip_3d) != 0:
                            # locate mid half way between left and right hips
                            hip_x = r_hip_3d[0] + ((l_hip_3d[0] - r_hip_3d[0]) / 2)
                            hip_y = r_hip_3d[1] + ((l_shoulder_3d[1] - r_hip_3d[1]) / 2)
                            hip_z = l_hip_3d[2] + ((r_hip_3d[2] - l_hip_3d[2]) / 2)
                            midhip_3d.append([hip_x, hip_y, hip_z])

                        else:
                            midhip_3d.append([np.nan] * 3)

                        lean_x, lean_y, lean_z = int(lm.x * frameWidth), int(lm.y * frameHeight), \
                                                 int(lm.z * frameWidth)

                        lean_2d.append([lean_x, lean_y])
                        lean_3d.append([lean_x, lean_y, lean_z])

                lean_2d = np.array(lean_2d, dtype=np.float64)
                lean_3d = np.array(lean_3d, dtype=np.float64)

                # PnP - different from the one for head as this is described to be more error-resistant
                # ==> see the point about hips above
                success2, rot_vec2, trans_vec2, inliers = cv2.solvePnPRansac(lean_3d, lean_2d, cam_matrix, dist_matrix)

                # Rotation matrix
                rotmat2, jac2 = cv2.Rodrigues(rot_vec2)

                # torso angles around axes
                angles2, mtxR2, mtxQ2, Qx2, Qy2, Qz2 = cv2.RQDecomp3x3(rotmat2)

                lean_x = angles2[0] * 360
                lean_y = angles2[1] * 360
                lean_z = angles2[2] * 360

                # adjust the values depending on calibration
                if lean_x > -5:
                    torso_text = "Back"
                elif lean_x < -20:
                    torso_text = "Forward"
                elif lean_y > 17:
                    torso_text = "Left"
                elif lean_y < 2:
                    torso_text = "Right"
                else:
                    torso_text = "Upright"

                if r_shoulder_dis > -192:
                    alter_lean = "Back"
                elif r_shoulder_dis < -480:
                    alter_lean = "Forward"
                else:
                    alter_lean = "Upright"

                # drawing shoulder projection
                l_shoulder_3d_projection, jacobian_l = cv2.projectPoints(l_shoulder_3d, rot_vec2, trans_vec2,
                                                                         cam_matrix, dist_matrix)
                r_shoulder_3d_projection, jacobian_r = cv2.projectPoints(r_shoulder_3d, rot_vec2, trans_vec2,
                                                                         cam_matrix, dist_matrix)
                # it seems like the projections above are not necessary
                # l_shoulder_3d projection can replace what l2 but the result is worse somehow...
                l1 = (int(l_shoulder_2d[0]), int(l_shoulder_2d[1]))
                l2 = (int(l_shoulder_2d[0] + lean_y * 10), int(l_shoulder_2d[1] - lean_x * 10))
                cv2.line(image, l1, l2, (255, 0, 255), 2)

                r1 = (int(r_shoulder_2d[0]), int(r_shoulder_2d[1]))
                r2 = (int(r_shoulder_2d[0] + lean_y * 10), int(r_shoulder_2d[1] - lean_x * 10))
                cv2.line(image, r1, r2, (128, 0, 128), 2)

                # add to pose rows
                lean_row = [np.round(lean_x, 6), np.round(lean_y, 6), np.round(lean_z, 6),
                            torso_text, alter_lean, neck_3d, midhip_3d]
                pose_row = pose_row + lean_row

                # draw on the video
                cv2.putText(image, "Torso: " + torso_text, (1600, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128),
                            2)
                cv2.putText(image, "Alternative: " + alter_lean, (1600, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (128, 0, 128),
                            2)
                cv2.putText(image, "x: " + str(np.round(lean_x, 2)), (1600, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                cv2.putText(image, "y: " + str(np.round(lean_y, 2)), (1600, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                cv2.putText(image, "z: " + str(np.round(lean_z, 2)), (1600, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
            else:
                pose_row = list(np.array([np.nan] * 34 * 4 + 4 + 3).flatten())
                # above will crash --- add up the numbers properly yourself ---
            # get hand information
            lefty_row = []
            if results.left_hand_landmarks:
                lefty = results.left_hand_landmarks.landmark
                lefty_row = list(
                    np.array([[int(landmark.x * frameWidth),
                               int(landmark.y * frameHeight),
                               int(landmark.z, *frameWidth)] for landmark in lefty]).flatten())
            else:
                lefty_row = list(np.array([np.nan] * 21 * 3).flatten())

            righty_row = []
            if results.right_hand_landmarks:
                righty = results.right_hand_landmarks.landmark
                righty_row = list(
                    np.array([[int(landmark.x * frameWidth),
                               int(landmark.y * frameHeight),
                               int(landmark.z * frameWidth)] for landmark in righty]).flatten())
            else:
                righty_row = list(np.array([np.nan] * 21 * 3).flatten())

            # below will be used in head pose estimation based on rotation around axes as well as for blinks
            face_2d = []
            face_3d = []
            face_row = []
            forehead = []
            if results.face_landmarks:
                face = results.face_landmarks.landmark
                ratio = blink_ratio(image, face)
                for idx, lm in enumerate(face):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:  # landmark numbers
                        if idx == 1:
                            nose_2d = (lm.x * frameWidth, lm.y * frameHeight)  # for later use in estimation
                            nose_3d = (lm.x * frameWidth, lm.y * frameHeight, lm.z * frameWidth)

                        head_x, head_y, head_z = int(lm.x * frameWidth), int(lm.y * frameHeight), \
                                                 int(lm.z * frameWidth)

                        face_2d.append([head_x, head_y])
                        face_3d.append([head_x, head_y, head_z])

                    if idx == 454:
                        coords = tuple(np.multiply(np.array((lm.x, lm.y)), [1920, 1080]).astype(int))

                    if idx == 9:
                        fhead_x, fhead_y, fhead_z = int(lm.x * frameWidth), int(lm.y * frameHeight), \
                                                    int(lm.z * frameWidth)
                        forehead.append = ([fhead_x, fhead_y, fhead_z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # PnP
                success1, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Rotational matrix
                rotmat, jac = cv2.Rodrigues(rot_vec)

                # Head angles around axes
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotmat)

                # Reverting from normalised values
                head_x = angles[0] * 360
                head_y = angles[1] * 360
                head_z = angles[2] * 360

                # the values below need to be adjusted depending on head\pose\camera position
                if head_x < -2:
                    text = "Down"
                elif head_x > 2:
                    text = "Up"
                elif head_y > 4:
                    text = "Left"
                elif head_y < 0:
                    text = "Right"
                else:
                    text = "Ahead"

                if head_z > 0.5 or head_z < -0.5:
                    text2 = "Tilt"
                else:
                    text2 = "No tilt"

                if text == "Ahead" and ratio > 2.7:
                    ratio2 = "Blinked"
                elif text == "Down" and ratio > 3:
                    ratio2 = "Blinked"
                else:
                    ratio2 = "Open"

                # writing rows
                face_row = [np.round(head_x, 6), np.round(head_y, 6), np.round(head_z, 6),
                            text, text2, np.round(ratio, 2), ratio2, forehead]

                # to draw nose projection and blinks on the image
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec,
                                                                 cam_matrix, dist_matrix)
                # the first output of the above can go into p2 definition like nose_3d_projection[0][0][0] for example
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + head_y * 50), int(nose_2d[1] - head_x * 50))
                cv2.line(image, p1, p2, (255, 0, 0), 2)

                cv2.putText(image, f"Eye openness : {np.round(ratio, 2)}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, ratio2, coords, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Head: " + text, (1600, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, text2, (1600, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "x: " + str(np.round(head_x, 2)), (1600, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(head_y, 2)), (1600, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(head_z, 2)), (1600, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                face_row = list(np.array([np.nan] * 10).flatten())

            # this gives world coordinates of wrists and hands in metres with mid hip being 0
            w_pose_row = []
            if results.world.pose.landmarks:
                w_pose = results.world_pose_landmarks.landmark
                # division by 100 to get values in cms
                w_pose_row2 = list(
                    np.array([[int(landmark.x / 100),
                               int(landmark.y / 100),
                               int(landmark.z / 100),
                               np.round(landmark.visibility, 6)] for landmark in w_pose]).flatten())

                # 60-67 left-right wrists | 76-83 left-right indexes
                w_pose_a = w_pose_row2[60:68] + w_pose_row2[76:84]
                w_pose_row.append(w_pose_a)

            else:
                w_pose_row = list(np.array([np.nan] * 16).flatten())

            # make a row to write to  the csv
            new_row = pose_row + lefty_row + righty_row + face_row + w_pose_row

            # insert time info
            new_row.insert(0, timee)

            # calculate fps info now that the main routine is done
            end = time.time()
            totalTime = end - start
            fps2 = 1 / totalTime

            # write FPS info
            cv2.putText(image, f'FPS: {int(fps2)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            vidout.write(image)  # write video with drawings

            # uncommenting below because docker doesn't have access to my display, so it can't show real-time video
            # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            timee += (1000 / samplerate)  # for the next iteration
            print(f"FPS: {np.round(fps2, 2)} at {timee / 1000} secs")

            # append row to the csv
            with open(csvoutput + ff[:-4] + '.csv', mode='a', newline='') as f2:
                csv_writer = csv.writer(f2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(new_row)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # to exit elegantly if cv2.imshow is uncommented
            break

    vidout.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f'Working on post-tracking processes of {ff}, started at {time.strftime("%H:%M:%S", time.localtime())}')
    # linear interpolation up to 500ms of missing data
    itp = pd.read_csv(csvoutput + ff[:-4] + '.csv')
    itp.interpolate(limit=25, inplace=True, limit_direction='both')

    # wrists are sliced to calculate whole arm velocity and acceleration
    # indexes are sliced to calculate hand velocity and acceleration
    # a landmark in the middle of the forehead is sliced to calculate head movement speed
    # alternatives are possible
    # sliced dataframes are named to create column names to be added to the original dataframe
    l_wrist = itp[["X_LEFT_WRIST", "Y_LEFT_WRIST", "Z_LEFT_WRIST"]]
    l_wrist.name = "L_WRIST"
    r_wrist = itp[["X_RIGHT_WRIST", "Y_RIGHT_WRIST", "Z_RIGHT_WRIST"]]
    r_wrist.name = "R_WRIST"
    l_index = itp[["X_INDEX_FINGER_TIP_L", "Y_INDEX_FINGER_TIP_L", "Z_INDEX_FINGER_TIP_L"]]
    l_index.name = "L_INDEX"
    r_index = itp[["X_INDEX_FINGER_TIP_R", "Y_INDEX_FINGER_TIP_R", "Z_INDEX_FINGER_TIP_R"]]
    r_index.name = "R_INDEX"
    fhead = itp[["FOREHEAD_X", "FOREHEAD_Y", "FOREHEAD_Z"]]
    fhead.name = "FHEAD"

    # calculating velocity and acceleration
    vlw = velo_acc_calc(l_wrist)
    vrw = velo_acc_calc(r_wrist)
    vli = velo_acc_calc(l_index)
    vri = velo_acc_calc(r_index)
    vf = velo_acc_calc(fhead)

    # adding these to the original dataframe
    itp2 = pd.DataFrame()
    itp2 = pd.concat([itp, vlw, vrw, vli, vri, vf], axis=1)

    # thumbs are sliced for gesture volume calculation with the index finger
    # l_thumb = itp[["X_THUMB_TIP_L", "Y_THUMB_TIP_L", "Z_THUMB_TIP_L"]]
    # l_thumb.name = "L_THUMB"
    # r_thumb = itp[["X_THUMB_TIP_R", "Y_THUMB_TIP_R", "Z_THUMB_TIP_R"]]
    # r_thumb.name = "R_THUMB"

    # writing a csv
    itp2.to_csv(csvoutput + ff[:-4] + '.csv', index=False, na_rep='nan')

    end_time = time.time()
    print(f'###########{ff} ended at {time.strftime("%H:%M:%S", time.localtime())}'
          f'and took {(end_time - start_time) / 60} mins ###########')

    # interpolation of NAs can be integrated into this
