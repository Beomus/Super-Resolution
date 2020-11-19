import argparse
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="path to model(s)")
ap.add_argument("-i", "--image", help="path to image(s)")
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

image = cv2.imread(args["image"])
print(f"[INFO] w : {image.shape[1]} | h: {image.shape[0]}")

start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print(f"[INFO] super resolution took {round(start-end, 2)} seconds.")
print(f"[INFO] w: {upscaled.shape[1]} | h: {upscaled.shape[0]}")

start = time.time()
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]),
            interpolation=cv2.INTER_AREA)

end = time.time()
print(f"[INFO] bicubic interpolation took {round(start - end, 2)} seconds.")

cv2.imshow("Original", image)
cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)
cv2.waitKey(0)
