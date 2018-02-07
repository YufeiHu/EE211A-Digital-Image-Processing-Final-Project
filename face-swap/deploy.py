from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
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


images_A = get_image_paths("data/trump")
images_B = get_image_paths("data/cage")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0


"""
cap = cv2.VideoCapture(0)


for epoch in range(100000):

    ret, frame = cap.read()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    offset = int((frame_width - frame_height) / 2)
    frame_ori = frame[:, offset : offset + frame_height, :]
    frame_test = cv2.resize(frame_ori, (64, 64), interpolation = cv2.INTER_AREA)
    frame_test = np.reshape(frame_test, (1, 64, 64, 3))


    frame_hat = autoencoder_A.predict(frame_test)
    frame_hat = np.clip(frame_hat[0]*255, 0, 255).astype('uint8')
    frame_hat = cv2.resize(frame_hat, (frame_height, frame_height), interpolation = cv2.INTER_AREA)


    figure = np.stack([frame_ori, frame_hat], axis=0)
    figure = stack_images(figure)


    cv2.imshow('frame', figure)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""



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
