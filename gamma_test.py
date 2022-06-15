import serial
import time
import numpy as np
from helper_main import *
from process_frame import process_frame
import torch
import cv2
from statespace import States

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


def compute_control_input(gamma_measure, gamma_reference):
    alpha_estimate = np.arcsin(n_pdms / n_eco * np.sin(gamma_measure))
    alpha_ref = np.arcsin(n_pdms / n_eco * np.sin(gamma_reference))
    alpha_error = alpha_ref - alpha_estimate
    if alpha_error > 0:
        return 1
    elif alpha_error < 0:
        return -1
    else:
        return 0


# get computing device (GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device ", device)

# load models for EE-dynamics
dx_dy_x_y_predictor = KinematicPredictor()
checkpoint = torch.load('./nn_models/kinematic_final1.pt',
                        map_location=torch.device('cpu'))  # adjust manually for best epoch
dx_dy_x_y_predictor.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters
dx_dy_x_y_predictor.to(device)
dx_dy_x_y_predictor.eval()

shape_predictor = ShapePredictor()
checkpoint = torch.load('./nn_models/shape_final1.pt',
                        map_location=torch.device('cpu'))  # adjust manually for best epoch
shape_predictor.load_state_dict(checkpoint['model_state_dict'])  # load saved parameters
shape_predictor.to(device)
shape_predictor.eval()

helper_air = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # helper to test NN's

cap = cv2.VideoCapture('./video_files/Zhiyuan.avi')

#  Define Parameters for Snell's law
n_eco = 1540
n_pdms = 1000

print("creating Arduino communication")
#arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)

print("getting camera indices")
#indices = get_camera_indices()  # get camera indices if 0, 1 don't work


# input
print("please input reference angle")
gamma_ref = float(input())

print("please press 'any key' to continue ")
start = input()

print("starting camera")
time.sleep(1)
cap_live = cap  # cv2.VideoCapture(2)  # watch out: index of second camera could be very high
# get first video frame

ok, frame = cap_live.read()

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.5, minDistance=10)

# convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use Shi-Tomasi to detect object corners / edges from initial frame
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shitomasi)

# create a black canvas the size of the initial frame
canvas = np.zeros_like(frame)
# create random colours for visualization for all 100 max corners for RGB channels
colours = np.random.randint(0, 255, (1000, 3))

# set min size of tracked object, e.g. 15x15px
parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# initialize states
state_space = States()


# main loop
while True:
    start = time.time()
    ok, frame = cap_live.read()
    if not ok:
        print("[INFO] end of file reached")
    gamma, frame_gray_init, edges = process_frame(frame, frame_gray_init, edges, parameter_lucas_kanade, canvas,
                                                  colours)

    print("gamma = ", gamma)
    delta_gamma = gamma_ref - gamma

    # helper_air, air_info = read_and_parse_data(arduino, state_space, device)
    # get vx, vy, x, x, and alpha estimation
    kinematics = dx_dy_x_y_predictor(helper_air).detach().numpy()
    state_space.update_states(kinematics[0, 2], kinematics[0, 3], 0.005)

    shape = shape_predictor(helper_air).detach().numpy()
    delta_x = compute_control_input(gamma_ref, gamma)
    #arduino.write(("<HelloWorld, 12, " + str(delta_x) + ">").encode())

    # output
    print("gamma_ref: ", gamma_ref, "Delta gamma: ", delta_gamma, "Sent delta_x =: ", delta_x, " to Serial Port")
    # print("air_info: ", air_info)
    time.sleep(0.07)

    if cv2.waitKey(20) == ord('q'):
        break

    end = time.time()
    loop_time = end - start
    print("elapsed time for one loop iteration: ", end - start)

"""Todo: Serial callback, for testing what arduino sends"""
