import cv2
import numpy as np

cap = cv2.VideoCapture('Zhiyuan.avi')
# get first video frame
ok, frame = cap.read()

# generate initial corners of detected object
# set limit, minimum distance in pixels and quality of object corner to be tracked
parameters_shitomasi = dict(maxCorners=100, qualityLevel=0.4, minDistance=3)

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

start_point = (0, 0)

# End coordinate
end_point = (200, 200)

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 9 px
thickness = 5

while True:
    # get next frame
    ok, frame = cap.read()
    image = cv2.arrowedLine(frame, start_point, end_point,
                            color, thickness)
    cv2.imshow('window', frame)
    if not ok:
        print("[INFO] end of file reached")
        # break
    # prepare grayscale image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # update object corners by comparing with found edges in initial frame
    update_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None,
                                                            **parameter_lucas_kanade)
    # only update edges if algorithm successfully tracked
    new_edges = update_edges[status == 1]
    # to calculate directional flow we need to compare with previous position
    old_edges = edges[status == 1]

    for i, (new, old) in enumerate(zip(new_edges, old_edges)):
        a, b = new.ravel()
        c, d = old.ravel()
        gamma = np.tan((a - b) / (c - d))

        # draw line between old and new corner point with random colour
        mask = cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), colours[i].tolist(), 2)
        # draw circle around new position
        frame = cv2.circle(frame, (int(a), int(b)), 5, colours[i].tolist(), -1)

    result = cv2.add(frame, mask)
    cv2.imshow('Optical Flow (sparse)', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # overwrite initial frame with current before restarting the loop
    frame_gray_init = frame_gray.copy()
    # update to new edges before restarting the loop
    edges = new_edges.reshape(-1, 1, 2)
