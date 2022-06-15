from pycromanager import Core
from pycromanager import Bridge
import numpy as np
import cv2

#Create the Micro-Managert to Pycro-Manager transfer layer
bridge = Bridge()
#get object representing micro-manager core
core = Core()
print(core)

while True:
    # tell the core to take a single image with the default camera
    core.snap_image()

    # retrive that image from the core. This will come in the form of a "Tagged image", which
    # is pixels + associated metadata
    tagged_image = core.get_tagged_image()
    #frame = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
    image_height = tagged_image.tags['Height']
    image_width = tagged_image.tags['Width']

    image = tagged_image.pix.reshape((image_height, image_width))
    cv2.imshow('a', image)

    if cv2.waitKey(20) == ord('q'):
        a=1

# pixels and metadata are returned as a NumPy ndarray, and a Python dict, respectively.
# print their types here to verify this
print(type(tagged_image.pix))
print((tagged_image.tags))

"""
while True:
    # tell the core to take a single image with the default camera
    core.snap_image()
    # retrieve that image from the core. This will come in the form of a "Tagged image", which
    # is pixels + associated metadata
    tagged_image = core.get_tagged_image()
    frame = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
"""
