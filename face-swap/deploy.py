from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
from image_augmentation import *
from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B
import numpy as np
import cv2


try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
except:
    pass


cap = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


for epoch in range(1000000):

    # Read original frame
    _, frame = cap.read()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    offset = int((frame_width - frame_height) / 2)
    frame = frame[:, offset : offset + frame_height, :]


    # Detect face
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame_gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
    

    # When at least a face is detected
    if len(faces) != 0:

        # Extract face patch
        face_x = faces[0][0]
        face_y = faces[0][1]
        face_w = faces[0][2]
        face_h = faces[0][3]
        face_patch = frame[face_y:face_y+face_h, face_x:face_x+face_w, :]
        if face_h < face_w:
            offset = int((face_w - face_h) / 2)
            face_patch = face_patch[:, offset : offset + face_h, :]
        else:
            offset = int((face_h - face_w) / 2)
            face_patch = face_patch[offset : offset + face_w, :, :]


        # Crop face patch
        offset_crop = face_patch.shape[0] // 10
        face_patch_crop = face_patch[offset_crop:face_patch.shape[0]-offset_crop, offset_crop:face_patch.shape[1]-offset_crop, :]
        frame_test = cv2.resize(face_patch_crop, (64, 64), interpolation = cv2.INTER_AREA)
        frame_test = frame_test / 255.0


        # Inference
        frame_hat = autoencoder_A.predict(frame_test.reshape((1, 64, 64, 3)))
        frame_hat = frame_hat[0, :, :, :]
        frame_hat = np.clip(frame_hat*255, 0, 255).astype('uint8')


        # Stitch
        face_stitch = cv2.resize(frame_hat, (face_patch_crop.shape[0], face_patch_crop.shape[1]), interpolation = cv2.INTER_AREA)
        frame_stitch = np.copy(frame)

        if face_h < face_w:
            x_start = face_x + offset_crop
            x_end = face_x + face_w - offset_crop
            y_start = face_y + offset + offset_crop
            y_end = face_y + offset + face_h - offset_crop
        else:
            x_start = face_x + offset + offset_crop
            x_end = face_x + offset + face_w - offset_crop
            y_start = face_y + offset_crop
            y_end = face_y + face_h - offset_crop
           
        frame_stitch[y_start:y_end, x_start:x_end, :] = face_stitch


        # Display the images
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
        frame_stitch = cv2.resize(frame_stitch, (512, 512), interpolation=cv2.INTER_AREA)
        face_patch_crop = cv2.resize(face_patch_crop, (512, 512), interpolation=cv2.INTER_AREA)
        frame_hat = cv2.resize(frame_hat, (512, 512), interpolation=cv2.INTER_AREA)

    else:

        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
        frame_stitch = np.zeros((512, 512, 3))
        face_patch_crop = np.zeros((512, 512, 3))
        frame_hat = np.zeros((512, 512, 3))


    figure1 = np.stack([frame, frame_stitch], axis=0)
    figure2 = np.stack([face_patch_crop, frame_hat], axis=0)
    figure = np.stack([figure1, figure2], axis=1)
    figure = stack_images(figure)

    cv2.imshow('frame', figure)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()