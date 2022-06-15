import serial
import time
import numpy as np
from helper_main import *
from process_frame import process_frame
import torch
import cv2
import logging
from hamamatsu.dcam import dcam, Stream, copy_frame, EventStream

from NN import KinematicPredictor, ShapePredictor


def read_and_parse_data(board, state, computing_device):
    available_bytes = board.in_waiting()
    print("available data bytes: ", available_bytes)
    if available_bytes > 0:
        uart_data = board.readline()
        air_data = list(extract_nums(uart_data))
        air_input_data = torch.tensor([[air_data[1], air_data[2], state.dist]], device=computing_device)

        return air_input_data, air_input_data.detach().cpu()

    return torch.tensor([0, 0, 0], device=computing_device), [0, 0, 0]


def compute_control_input(gamma_measure, gamma_reference, n_eco, n_pdms):
    alpha_estimate = np.arcsin(n_pdms / n_eco * np.sin(gamma_measure))
    alpha_ref = np.arcsin(n_pdms / n_eco * np.sin(gamma_reference))
    alpha_error = alpha_ref - alpha_estimate
    if alpha_error > 0:
        return 1
    elif alpha_error < 0:
        return -1
    else:
        return 0

#  Define Parameters for Snell's law
n_eco = 1540
n_pdms = 1000

# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.5, minDistance=10)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (1000, 3))
# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


print("creating Arduino communication")
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)
print("getting camera indices")
# indices = get_camera_indices()  # get camera indices if 0, 1 don't work
# input
print("please input reference angle")
gamma_ref = float(input())
print("please press 'any key' to continue ")
start = input()
print("starting camera")
time.sleep(1)

# main loop
logging.basicConfig(level=logging.INFO)

with dcam:
    camera = dcam[0]
    with camera:
        print(camera.info)
        print(camera['image_width'].value, camera['image_height'].value)

        # Simple acquisition example
        nb_frames = 10000
        isFirstFrame = 1
        camera["exposure_time"] = 0.01
        with Stream(camera, nb_frames) as stream:
            logging.info("start acquisition")
            camera.start()
            for i, frame_buffer in enumerate(stream):
                if isFirstFrame:
                    # generate initial corners of detected object
                    # convert to grayscale
                    frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Use Shi-Tomasi to detect object corners / edges from initial frame
                    edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shitomasi)
                    # create a black canvas the size of the initial frame
                    canvas = np.zeros_like(frame)
                    isFirstFrame = 0

                frame = copy_frame(frame_buffer)
                gamma, frame_gray_init, edges = process_frame(frame, frame_gray_init, edges, parameter_lucas_kanade,
                                                              canvas,
                                                              colours)

                print("gamma = ", gamma)
                delta_gamma = gamma_ref - gamma
                delta_x = compute_control_input(gamma_ref, gamma, n_eco, n_pdms)
                arduino.write(("<HelloWorld, 12, " + str(delta_x) + ">").encode())
                # output
                print("gamma_ref: ", gamma_ref, "Delta gamma: ", delta_gamma, "Sent delta_x =: ", delta_x,
                      " to Serial Port")
                time.sleep(0.07)

                if cv2.waitKey(20) == ord('q'):
                    break

                logging.info(f"acquired frame #%d/%d: %s", i + 1, nb_frames, frame)
            logging.info("finished acquisition")
