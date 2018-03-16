from utils import get_image_paths, load_images, stack_images
import numpy as np
import cv2


file = open("dataset/yufei/log.txt","r")
num = file.readlines()
num = int(num[0])
file.close()

cap = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(cascPath)
for epoch in range(100000):
    
    ret, frame = cap.read()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    offset = int((frame_width - frame_height) / 2)
    frame = frame[:, offset : offset + frame_height, :]
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
        
    frame_resize = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_AREA)
    figure = np.stack([frame_resize, face_patch], axis=0)
    figure = stack_images(figure)
    cv2.imshow('frame', figure)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    elif ch == ord('c'):
        savePath = "dataset/yufei/y" + str(num) + '.png'
        cv2.imwrite(savePath, face_patch)
        num += 1


cap.release()
cv2.destroyAllWindows()

file1 = open("dataset/yufei/log.txt","w")
file1.write(str(num))
file1.close()