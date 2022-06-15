import logging
from hamamatsu.dcam import dcam, Stream, copy_frame, EventStream


logging.basicConfig(level=logging.INFO)

with dcam:
    camera = dcam[0]
    with camera:
        print(camera.info)
        print(camera['image_width'].value, camera['image_height'].value)

        # Simple acquisition example
        nb_frames = 10
        camera["exposure_time"] = 0.1
        with Stream(camera, nb_frames) as stream:
            logging.info("start acquisition")
            camera.start()
            for i, frame_buffer in enumerate(stream):
                frame = copy_frame(frame_buffer)
                logging.info(f"acquired frame #%d/%d: %s", i + 1, nb_frames, frame)
            logging.info("finished acquisition")

with dcam:
    camera = dcam[0]
    with camera:
        print(camera.info)
        print(camera['image_width'].value, camera['image_height'].value)

        # Simple acquisition example
        nb_frames = 10
        camera["exposure_time"] = 0.1
        with EventStream(camera) as stream:
            logging.info("start acquisition")
            camera.start()
            for i, frame_buffer in enumerate(stream):
                frame = copy_frame(frame_buffer)
                logging.info(f"acquired frame #%d/%d: %s", i + 1, nb_frames, frame)
            logging.info("finished acquisition")
