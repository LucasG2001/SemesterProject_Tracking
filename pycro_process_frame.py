import cv2
import numpy as np


def update_gamma_estimate(motions):
    x_diff = np.mean(motions[:, 0])
    y_diff = np.mean(motions[:, 1]) + 0.0000001  # trick to approximate tan without worrying about division through 0 without if-statement
    gamma = (np.tan(x_diff / y_diff))

    return gamma, int(600 + 1000 * x_diff), int(1000 * y_diff + 600)


def set_dummy_values(canvas, colours, frame):
    new_edges = np.array([[1, 1], [1, 1]]).astype('float32')
    old_edges = np.array([[0, 0], [0, 0]]).astype('float32')
    motions = np.array([[0, 0]])
    a, b = 1, 1
    c, d = 0, 0
    # draw line between old and new corner point with random colour
    mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[0].tolist(), 2)
    # draw circle around new position
    frame = cv2.circle(frame, (int(a), int(b)), 5, colours[0].tolist(), -1)

    return new_edges, old_edges, motions, mask, frame


def check_validity(edges_new, edges_old):
    if edges_new.size == 0 or edges_old.size == 0:
        return 0
    else:
        return 1


def process_frame(frame, frame_gray_init, edges, parameter_lucas_kanade, canvas, colours):
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame, edges, None,
                                                            **parameter_lucas_kanade)
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges[status == 1]
    # Test case: No edges can be tracked, but we want to see if the rest of the algo runs
    if check_validity(new_edges, old_edges) == 0:
        new_edges, old_edges, motions, mask, frame = set_dummy_values(canvas, colours, frame)
    else:
        motions = np.subtract(new_edges, old_edges)

    gamma, x_motion, y_motion = update_gamma_estimate(motions)

    # for the line in the video showing the measured direction
    frame = cv2.arrowedLine(frame, pt1=(600, 600), pt2=(x_motion, y_motion), color=(0, 255, 0), thickness=5)

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
    frame_gray_init = frame.copy()
    # update to new edges before restarting the loop
    edges = new_edges.reshape(-1, 1, 2)

    return gamma, frame_gray_init, edges
