import numpy as np
import time
import cv2
import torch

def convert_to_8bit(frame):
    intermediate_res = frame * (255/np.amax(frame))
    return intermediate_res.astype('uint8')

"""function to extract floats from comma separated text """
def extract_nums(text):
    for item in text.split(b','):
        try:
            yield float(item)
        except ValueError:
            pass

def setup_cv_parameters():
    # set limit, minimum distance in pixels and quality of object corner to be tracked
    parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.1, minDistance=1)
    # create random colours for visualization for all 100 max corners for RGB channels
    colours = np.random.randint(0, 255, (1000, 3))
    # set min size of tracked object, e.g. 15x15px
    parameter_lucas_kanade = dict(winSize=(30, 30), maxLevel=2,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    return parameters_shitomasi, parameter_lucas_kanade, colours

def get_initial_frame(core, shitomasi_param):
    core.snap_image()
    tagged_image = core.get_tagged_image()
    frame = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
    frame = convert_to_8bit(frame)
    # Use Shi-Tomasi to detect object corners / edges from initial frame
    edges = cv2.goodFeaturesToTrack(frame, mask=None, **shitomasi_param)
    # create a black canvas the size of the initial frame
    canvas = np.zeros_like(frame)

    return frame, edges, canvas


def get_camera_indices():
    indices = []
    for i in range(10000):
        cap = cv2.VideoCapture(i)
        if (cap.isOpened()):
            indices.append(i)

    print("possible camera indices: ", indices)
    return indices


def measure_gamma(gamma_ref, error):
    gamma = gamma_ref + error
    print("Detected bubbles moving in direction: ", gamma)
    return gamma


def compute_dx(d_alpha, a=0.5, b=15, c=200):
    sin = np.sin(d_alpha)
    cos = np.cos(d_alpha)
    A = 4 * a * a * sin * sin
    B = -4 * a * b * (1 + cos * cos)
    C = b * b - sin * sin - cos * cos

    if (B * B - 4 * A * C) < 0:
        x1 = 0
        x2 = 0
    else:
        x1 = (-B - np.sqrt(B * B - 4 * A * C)) / 2 * A
        x2 = (-B + np.sqrt(B * B - 4 * A * C)) / 2 * A
    if x1 < x2:
        return x1
    else:
        return x2


def compute_y(a, b, c, x):
    return a * x * x + b * x + c


def write_read(x, arduino):
    arduino.write(bytes(str(x), 'utf-8'))
    data = arduino.readline()
    return data

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