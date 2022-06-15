import cv2
import numpy as np

start = (600, 600)

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 9 px
thickness = 5

def update_gamma_estimate(motions):
    count = 0
    x_diff = np.mean(motions[:, 0])
    y_diff = np.mean(motions[:, 1])
    gamma = (np.tan(x_diff / y_diff))

    return gamma, int(600 + 1000 * x_diff), int(1000 * y_diff + 600)


def process_frame(frame, frame_gray_init, edges, parameter_lucas_kanade, canvas, colours):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None,
                                                            **parameter_lucas_kanade)
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges[status == 1]
    motions = np.subtract(new_edges, old_edges)
    gamma, x_motion, y_motion = update_gamma_estimate(motions)
    end = (x_motion, y_motion)
    frame = cv2.arrowedLine(frame, start, end,
                            color, thickness)
    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()

        # draw line between old and new corner point with random colour
        mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[i].tolist(), 2)
        # draw circle around new position
        frame = cv2.circle(frame, (int(a), int(b)), 5, colours[i].tolist(), -1)

    result = cv2.add(frame, mask)
    cv2.imshow('Optical Flow (sparse)', result)
    # overwrite initial frame with current before restarting the loop
    frame_gray_init = frame_gray.copy()
    # update to new edges before restarting the loop
    edges = new_edges.reshape(-1, 1, 2)

    return gamma, frame_gray_init, edges
