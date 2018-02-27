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
    ret, frame = cap.read()
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
    
    face_patch = np.zeros((256, 256, 3), dtype=np.uint8)
    if len(faces) != 0:
        face_x = faces[0][0]
        face_y = faces[0][1]
        face_w = faces[0][2]
        face_h = faces[0][3]
        face_patch = frame[face_y:face_y+face_h, face_x:face_x+face_w, :]
        patch_height = face_patch.shape[0]
        patch_width = face_patch.shape[1]
        if patch_height < patch_width:
            offset = int((patch_width - patch_height) / 2)
            face_patch = face_patch[:, offset : offset + patch_height, :]
        else:
            offset = int((patch_height - patch_width) / 2)
            face_patch = face_patch[offset : offset + patch_width, :, :]
        face_patch = cv2.resize(face_patch, (256, 256), interpolation = cv2.INTER_AREA)
        cv2.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)

    frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_AREA)


    # Random crop testing frame
    _, face_patch_warp = random_warp(face_patch)
    frame_test = face_patch_warp / 255.0
    #print(frame_test.shape)
    #print(frame_test[0:50, 0, 0])
    face_patch_warp = cv2.resize(face_patch_warp, (256, 256), interpolation = cv2.INTER_AREA)


    # Inference
    frame_hat = autoencoder_A.predict(frame_test.reshape((1, 64, 64, 3)))
    frame_hat = frame_hat[0, :, :, :]
    
    frame_hat = np.clip(frame_hat*255, 0, 255).astype('uint8')
    frame_hat = cv2.resize(frame_hat, (256, 256), interpolation = cv2.INTER_AREA)
    #print(frame_hat.shape)
    #print(frame[0:50, 0, 0])
    #print(face_patch[0:50, 0, 0])
    #print(frame_hat[0:50, 0, 0])
    #print('\n')


    # Show the result
    figure = np.stack([frame, face_patch_warp, frame_hat], axis=0)
    figure = stack_images(figure)

    cv2.imshow('frame', figure)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



"""
images_A = get_image_paths("data/trump") # Qi
images_B = get_image_paths("data/cage") # Yufei
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))


test_trump = images_A[100, :, :, :]
test_cage = images_B[100, :, :, :]
test_trump = cv2.resize(test_trump, (64, 64), interpolation = cv2.INTER_AREA)
test_cage = cv2.resize(test_cage, (64, 64), interpolation = cv2.INTER_AREA)
test_trump = test_trump.reshape((1, test_trump.shape[0], test_trump.shape[1], test_trump.shape[2]))
test_cage = test_cage.reshape((1, test_cage.shape[0], test_cage.shape[1], test_cage.shape[2]))
print(test_trump.shape)
print(test_cage.shape)


hat_trump_A = autoencoder_A.predict(test_trump)
hat_cage_A = autoencoder_A.predict(test_cage)
print(hat_trump_A.shape)
print(hat_cage_A.shape)


hat_trump_B = autoencoder_B.predict(test_trump)
hat_cage_B = autoencoder_B.predict(test_cage)
print(hat_trump_B.shape)
print(hat_cage_B.shape)


figure_A = np.stack([test_trump, hat_trump_A, hat_trump_B], axis=0)
figure_B = np.stack([test_cage, hat_cage_A, hat_cage_B], axis=0)
figure = np.concatenate([figure_A, figure_B], axis=0)
figure = figure.reshape((2, 3) + figure.shape[1:])
figure = stack_images(figure)
figure = np.clip(figure*255, 0, 255).astype('uint8')


for epoch in range(1000000):
	cv2.imshow("", figure)
	key = cv2.waitKey(1)
	if key == ord('q'):
		exit()
"""