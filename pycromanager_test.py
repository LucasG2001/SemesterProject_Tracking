from pycromanager import Core
from pycromanager import Bridge
import numpy as np
import serial
import time
from helper_main import setup_cv_parameters, get_initial_frame, compute_control_input, convert_to_8bit
from pycro_process_frame import process_frame
import cv2
import logging

def get_frame(core_MM2):
    # tell the core to take a single image with the default camera
    core_MM2.snap_image()
    # retrieve that image from the core. This will come in the form of a "Tagged image", which
    # is pixels + associated metadata
    tagged_image = core.get_tagged_image()
    image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
    image = convert_to_8bit(image)
    return image

#  Define Parameters for Snell's law and tracking algorithms
n_eco = 1540
n_pdms = 1000

parameters_shitomasi, parameter_lucas_kanade, colours = setup_cv_parameters()

print("creating Arduino communication")
#arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)
# input
print("please input reference angle")
gamma_ref = float(input())
print("please press 'any key' to continue ")
start = input()
print("starting camera")
time.sleep(1)

# Create the Micro-Managert to Pycro-Manager transfer layer
bridge = Bridge()
# get object representing micro-manager core
core = Core()


frame_gray_init, edges, canvas = get_initial_frame(core, parameters_shitomasi)

while True:
    frame = get_frame(core)
    gamma, frame_gray_init, edges = process_frame(frame, frame_gray_init, edges, parameter_lucas_kanade,
                                                  canvas,
                                                  colours)
    print("gamma = ", gamma)
    delta_gamma = gamma_ref - gamma
    delta_x = compute_control_input(gamma_ref, gamma, n_eco, n_pdms)
    #arduino.write(("<HelloWorld, 12, " + str(delta_x) + ">").encode())
    # output
    print("gamma_ref: ", gamma_ref, "Delta gamma: ", delta_gamma, "Sent delta_x =: ", delta_x,
          " to Serial Port")
    time.sleep(0.07)

    if cv2.waitKey(20) == ord('q'):
        break
