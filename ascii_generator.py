import math
import cv2                                      # python -m pip install opencv-python
import numpy as np                              # python -m pip install numpy
import time
import multiprocessing as mp
from tqdm import trange
# from functools import partial
from PIL import Image, ImageDraw, ImageFont


ascii_order = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


def find_char(gray):
    return ascii_order[math.floor(gray * (len(ascii_order) / 256))]


def img_rgb_to_ascii(img, fontname, scale, char_width, char_height, fontsize):
    w, h = img.size
    img = img.resize(
        (int(scale * w), int(scale * h * (char_width / char_height))), Image.ANTIALIAS)
    w, h = img.size

    font = ImageFont.truetype("fonts\\%s.ttf" % fontname, fontsize)

    output_img = Image.new("RGB", (w * char_width, h * char_height), (0, 0, 0))
    d = ImageDraw.Draw(output_img)

    # img_rgb = np.asarray(img)

    # for i in range(h):
    #     for j in range(w):
    #         rgb = img_rgb[i, j]
    #         gray = sum(rgb)/3
    #         d.text((j * char_width, i * char_height),
    #                find_char(gray), font=font, fill=tuple(rgb))

    img_rgb = img.load()

    for i in range(h):
        for j in range(w):
            rgb = img_rgb[j, i]
            gray = sum(rgb) / 3
            d.text((j * char_width, i * char_height),
                   find_char(gray), font=font, fill=rgb)

    return output_img


def convert_img_from_path(image_path, output_file="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    t1 = time.time()

    img = Image.open(image_path).convert("RGB")

    output_img = img_rgb_to_ascii(img, fontname, scale,
                                  char_width, char_height, fontsize)
    output_img.save("%s.png" % output_file)

    t2 = time.time()

    print("Image %s.png has been successfully generated from %s, completed in %f seconds" %
          (output_file, image_path, round(t2-t1, 2)))


def vid_to_ascii(frame_array, fontname, scale, char_width, char_height, fontsize):
    t1 = time.time()

    output_frame_array = []

    # for f in frame_array:
    #     f = Image.fromarray(f)
    #     output_frame_array.append(img_rgb_to_ascii(
    #         f, fontname, scale, char_width, char_height, fontsize))

    for i in trange(len(frame_array)):
        f = Image.fromarray(frame_array[i])
        output_frame_array.append(img_rgb_to_ascii(
            f, fontname, scale, char_width, char_height, fontsize))

    t2 = time.time()

    print(f"Images to ASCII took {round(t2-t1,2)} seconds.")

    return output_frame_array


def convert_vid_from_path(video_path, output_file="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    t1 = time.time()

    vid = cv2.VideoCapture(video_path)

    frame_array = []

    # Can't use multiprocessing, or at least I don't think you can
    for f in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, img_bgr = vid.read()
        try:
            frame_array.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        except:
            break

    output_frame_array = vid_to_ascii(
        frame_array, fontname, scale, char_width, char_height, fontsize)

    w, h = output_frame_array[0].size

    fps = vid.get(cv2.CAP_PROP_FPS)
    output_vid = cv2.VideoWriter(
        "%s.mp4" % output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in range(len(output_frame_array)):
        try:
            output_vid.write(cv2.cvtColor(
                np.array(output_frame_array[i]), cv2.COLOR_RGB2BGR))
        except:
            output_vid.release()
            break

    cv2.destroyAllWindows()
    vid.release()
    output_vid.release()

    t2 = time.time()
    print("Video %s.mp4 has been successfully generated from %s, completed in %d seconds." %
          (output_file, video_path, round(t2-t1, 2)))


def convert_vid_from_dir(dir, output_dir="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    pass


start = time.time()

# print(find_char(231))
convert_img_from_path("3185f4b8-d3c2-4dbe-ae35-9c5033fbf856.jpg", scale=0.16)
# convert_vid_from_path("Glowing_Open_Fire__Videvo.mov", scale=0.08)

end = time.time()
print(f"Total Elapsed: {round(end-start,2)} seconds")