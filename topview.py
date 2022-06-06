import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import csv
import time
from os import listdir
from os.path import isfile, join

# prepare folders for looping
videoinput = "./video_input/topview/"
eachfile = [f for f in listdir(videoinput) if isfile(join(videoinput, f))]
csvoutput = "./csv_output/topview/"
videooutput = "./video_output/topview/"

# mediapipe solutions to be used
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Creating columns
markerz = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
           'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
           'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
           'RING_MCP', 'RING_TIP', 'RING_DIP', 'RING_TIP',
           'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

markerxyz = []
for mark in markerz:
    for pos in ['X', 'Y', 'Z']:
        nm = pos + "_" + mark
        markerxyz.append(nm)

time_row = ['TIME', 'FRAME', 'INDEX', 'CONFIDENCE', 'HAND']  # the last four will be extracted in the function below
whole_row = time_row + markerxyz

# it turns the landmark coordinates and handedness results are stored separately
# this means that coordinates are not associated with any hands directly
# below tries to take care of that but there might be a better solution

def get_label(index, hand, results):
    output = None
    for classification in results.multi_handedness:
        if index == 0:
            inndex = 0
            label = results.multi_handedness[0].classification[0].label
            score = results.multi_handedness[0].classification[0].score
            # these are to draw handedness on the image. Not necessary
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))

            output = text, coords, label, score, inndex
            return output

        if index == 1:
            inndex = 1
            label = results.multi_handedness[1].classification[0].label
            score = results.multi_handedness[1].classification[0].score
            # these are to draw handedness on the image. Not necessary
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))

            output = text, coords, label, score, inndex
            return output

        if index == 2:
            inndex = 2
            label = results.multi_handedness[2].classification[0].label
            score = results.multi_handedness[2].classification[0].score
            # these are to draw handedness on the image. Not necessary
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))

            output = text, coords, label, score, inndex
            return output

        if index == 3:
            inndex = 3
            label = results.multi_handedness[3].classification[0].label
            score = results.multi_handedness[3].classification[0].score
            # these are to draw handedness on the image.
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [1920, 1080]).astype(int))

            output = text, coords, label, score, inndex
            return output


for ff in eachfile:
    start_time = time.time()
    print(f'###########{ff} started at {time.strftime("%H:%M:%S", time.localtime())}###########')

    # initialise a csv
    with open(csvoutput + ff[:-4] + '.csv', mode='w', newline='') as cs:
        csv_writer = csv.writer(cs, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(whole_row)

    # Read video frames
    cap = cv2.VideoCapture(videoinput + ff)
    timee = 0  # this will contain time information in the loop
    frame = 1
    # extract video properties
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # prepare video output with drawings
    samplerate = fps  # may reduce quality for speed
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # or maybe (*'XVID')
    vidout = cv2.VideoWriter(videooutput + ff[:-4] + '.mp4', fourcc, fps=samplerate,
                             frameSize=(int(frameWidth), int(frameHeight)))

    # main routine
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=4,
                        min_detection_confidence=0.5,  # can be increased
                        min_tracking_confidence=0.5,
                        model_complexity=1) as hands:
        while cap.isOpened():
            success, image = cap.read()
            start = time.time()  # curious how long the processing will take on cpu

            if not success:
                print("No frame could be read")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MP requirement

            # handedness detection seems to rely on flipped images, which is not very accurate in the first place.
            # No need to do this because the explainer is not in selfie view
            # image = cv2.flip(image, 1)

            image.flags.writeable = False  # improves performance

            results = hands.process(image)  # main container of data

            # print(results.pose_landmarks)

            # Draw landmarks on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            handies_row = []
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(51, 255, 255),
                                                                     thickness=3, circle_radius=3),
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                     circle_radius=1))
                    # get handedness rows
                    hand_info = []
                    if get_label(num, hand, results):
                        text, coords, label, score, inndex = get_label(num, hand, results)
                        cv2.putText(image, text, coords, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)

                        hand_info.append(timee)
                        hand_info.append(frame)
                        hand_info.append(inndex)
                        hand_info.append(np.round(score, 6))
                        hand_info.append(label)

                    else:
                        hand_info = list(np.array([np.nan] * 3).flatten())
                        hand_info.insert(0, timee)
                        hand_info.insert(1, frame)

                    # get hand coordinates
                    handies = hand.landmark
                    handies_row = list(
                        np.array([[int(landmark.x * frameWidth),
                                   int(landmark.y * frameHeight),
                                   np.round(landmark.z, 6)] for landmark in handies]).flatten())  # 21 * 3 rows

            else:
                handies_row = list(np.array([np.nan] * 21 * 3).flatten())

            handies_row2 = hand_info + handies_row

            # calculate fps info
            end = time.time()
            totalTime = end - start
            fps2 = 1 / totalTime

            # write FPS info
            cv2.putText(image, f'FPS: {int(fps2)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            vidout.write(image)  # write video with drawings

            # no display with the following line due to docker and qt gui issue
            # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

            # for the next iteration
            timee = round(timee + (1000 / samplerate))
            frame = frame + 1
            print(f"FPS: {np.round(fps2, 2)}, at {timee / 1000} seconds")  # on the terminal

            # append row to the csv
            with open(csvoutput + ff[:-4] + '.csv', mode='a', newline='') as f2:
                csv_writer = csv.writer(f2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(handies_row2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vidout.release()
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f'###########{ff} ended at {time.strftime("%H:%M:%S", time.localtime())}'
          f'and took {(end_time - start_time) / 60} mins ###########')

    # interpolation of NAs can be integrated into this
