from utils import get_image_paths, load_images, stack_images
import numpy as np
import cv2


cap = cv2.VideoCapture(0)


for epoch in range(100000):

    ret, frame = cap.read()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    offset = int((frame_width - frame_height) / 2)
    frame_test = frame[:, offset : offset + frame_height, :]
    frame_test = cv2.resize(frame_test, (64, 64), interpolation = cv2.INTER_AREA)

    figure = np.stack([frame_test, frame_test], axis=0)
    figure = stack_images(figure)

    # Display the resulting frame
    cv2.imshow('frame', figure)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()