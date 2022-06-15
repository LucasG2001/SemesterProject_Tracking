import serial
import time
import numpy as np
from helper_main import*
import cv2

"""Define Parameters for Snell's law"""
n_ecoflex = 100
n_fluid = 100

arduino = serial.Serial(port='COM7', baudrate=9600, timeout=.1)

# input
#main loop
gamma_ref = int(input())
error = 12

while True:
    gamma = measure_gamma(gamma_ref, error)
    delta_gamma = gamma_ref - gamma
    delta_alpha = np.arcsin(n_fluid / n_ecoflex * np.sin(delta_gamma))
    delta_x = compute_dx(delta_alpha)
    air_info = write_read("<HelloWorld, 12, " + str(delta_x) + " >", arduino)
    # output
    print("gamma_ref: ", gamma_ref, "Delta gamma: ", delta_gamma, "Sent delta_x =: ", delta_x, " to Serial Port")
    print("air_info: ", air_info)
    time.sleep(0.25)
    error = error - 0.35

"""Todo: Serial callback, for testing what arduino sends"""