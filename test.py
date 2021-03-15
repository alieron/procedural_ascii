import numpy as np
import cv2
from PIL import Image

# image = np.array([[[1, 2, 2],
#                    [3, 2, 2],
#                    [3, 3, 3]],
#                   [[2, 7, 4],
#                    [7, 2, 9],
#                    [7, 4, 3]]])
# print(image)

# convert = image[:, :, ::-1]
# print(convert)

# h, w, _ = image.shape
# print(w, h)

# for i in range(h):
#     for j in range(w):
#         rgb = image[i, j]
#         gray = np.sum(rgb)
#         print(rgb, gray)

# print(gray)

# image = np.asarray(cv2.imread("Pilot Wallpaper.png"))
# print(image, type(image))

image = Image.open("Pilot Wallpaper Square2.png")
w, h = image.size
print(w, h)
image = image.resize((w*2, h*2))
w, h = image.size
print(w, h)
image.save("resized.png")
