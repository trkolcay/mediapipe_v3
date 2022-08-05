import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import csv
import time
from os import listdir
from os.path import isfile, join

# folder path
soundinput = "./sound_input/"
soundoutput = "./sound_output/f0_intensity/"
inter_pol = "./sound_output/f0_intensity/interpolated/"
eachsound = [q for q in listdir(soundinput) if isfile(join(soundinput, q))]

# this is to extract pitch and intensity aligned to the video framerate
for qq in eachsound:
    start_time = time.time()
    print(f'Working on {qq} at {time.strftime("%H:%M:%S", time.localtime())}')

    # initialise a csv with headers
    columns = ["Hz", "dB"]
    with open(soundoutput + qq[:-4] + '.csv', mode='w', newline='') as cs3:
        csv_writer = csv.writer(cs3, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(columns)

    # reading the sound file
    wavfile = parselmouth.Sound(soundinput + qq)

    # these are praat commands from parselmouth
    tmin = call(wavfile, 'Get start time')
    tmax = call(wavfile, 'Get end time')

    # F0 and Intensity readings cannot produce output at the same times.
    # This is due to a difference in framing in the praat code.
    # So I will get high resolution sampling with a timestep of 1ms and then interpolate
    tointens = call(wavfile, 'To Intensity', 75, 0.001, True)
    topitch = call(wavfile, 'To Pitch', 0.001, 75, 600)

    # Now extract the readings at every 20ms only so that it aligns with the video data
    for i in np.arange(tmin, tmax, 0.02):
        intensity = call(tointens, "Get value at time", i, 'cubic')
        pitch = call(topitch, 'Get value at time', i, 'Hertz', 'Linear')
        columns2 = [pitch, intensity]
        with open(soundoutput + qq[:-4] + '.csv', mode='a', newline='') as cs4:
            csv_writer = csv.writer(cs4, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(columns2)

    print(f'Interpolating {qq}')
    # linear interpolation
    itp = pd.read_csv(soundoutput + qq[:-4] + '.csv')
    itp.interpolate(limit=12, inplace=True, limit_direction='both')
    itp = np.round(itp, 1)
    itp.to_csv(inter_pol + qq[:-4] + '.csv', index=False, na_rep='nan')

    end_time = time.time()
    duration = end_time - start_time
    print(f'Completed {qq} at {time.strftime("%H:%M:%S", time.localtime())} and it took {np.round((duration / 60), 2)} mins')
