import imutils
from imutils.video import VideoStream
import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="path to model")
args = vars(ap.parse_args())

modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

print(f"[INFO] loading model {args['model']}")
print(f"[INFO] model name: {modelName}")
print(f"[INFO] model scale: {modelScale}")

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=300)

    upscaled = sr.upsample(frame)
    bicubic = cv2.resize(frame,
                (upscaled.shape[1], upscaled.shape[0]),
                interpolation=cv2.INTER_CUBIC)
    
    cv2.imshow("Original", frame)
    cv2.imshow("Bicubic", bicubic)
    cv2.imshow("Super Resolution", upscaled)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

