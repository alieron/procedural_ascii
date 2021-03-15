import math
import cv2                                      # python -m pip install opencv-python
import numpy as np                              # python -m pip install numpy
import time
import multiprocessing as mp
from functools import partial
from PIL import Image, ImageDraw, ImageFont


ascii_order = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


def find_char(gray):
    return ascii_order[math.floor(gray * (len(ascii_order) / 256))]


def img_rgb_to_ascii(img, fontname, scale, char_width, char_height, fontsize):
    w, h = img.size
    img = img.resize(
        (int(scale * w), int(scale * h * (char_width / char_height))), Image.NEAREST)
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
    img = Image.open(image_path).convert("RGB")

    t7 = time.time()
    output_img = img_rgb_to_ascii(img, fontname, scale,
                                  char_width, char_height, fontsize)
    output_img.save("%s.png" % output_file)
    t8 = time.time()

    print(t8-t7)
    print("Image %s.png has been successfully generated from %s" %
          (output_file, image_path))


def vid_to_ascii(frame_array, fontname, scale, char_width, char_height, fontsize):
    output_frame_array = []

    t5 = time.time()
    for f in frame_array:
        f = Image.fromarray(f)
        output_frame_array.append(img_rgb_to_ascii(
            f, fontname, scale, char_width, char_height, fontsize))

    # TODO: Maybe figure out multiprocessing?
    # processes = []
    # for f in frame_array:
    #     p = mp.Process(target=img_rgb_to_ascii, args=(
    #         frame_array, fontname, scale, char_width, char_height, fontsize))
    #     processes.append(p)
    #     p.start()
    # for p in processes:
    #     p.join()

    # pool = mp.Pool(processes=4)
    # img_process = partial(img_rgb_to_ascii, fontname=fontname, scale=scale,
    #                       char_width=char_width, char_height=char_height, fontsize=fontsize)
    # output_frame_array = pool.map(img_process, frame_array)

    t6 = time.time()

    print(f"Images to ASCII took {t6-t5} seconds")

    return output_frame_array


def convert_vid_from_path(video_path, output_file="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    t1 = time.time()

    vid = cv2.VideoCapture(video_path)

    frame_array = []

    t1 = time.time()

    # Can't use multiprocessing, or at least I don't think you can
    for f in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, img_bgr = vid.read()
        frame_array.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    t2 = time.time()

    print(f"BGR2RGB took {t2-t1} seconds")

    output_frame_array = vid_to_ascii(
        frame_array, fontname, scale, char_width, char_height, fontsize)

    w, h = output_frame_array[0].size
    fps = vid.get(cv2.CAP_PROP_FPS)
    output_vid = cv2.VideoWriter(
        "%s.mp4" % output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    t3 = time.time()
    # TODO: RGB2BGR potential speed up using numpy & multiprocessing/multithreading
    for i in range(len(output_frame_array)):
        output_vid.write(cv2.cvtColor(
            np.array(output_frame_array[i]), cv2.COLOR_RGB2BGR))
    t4 = time.time()

    cv2.destroyAllWindows()
    vid.release()
    output_vid.release()

    print(f"RGB2BGR took {t4-t3} seconds")

    print("Video %s.mp4 has been successfully generated from %s" %
          (output_file, video_path))


def convert_vid_from_dir(dir, output_dir="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    pass


start = time.time()

# print(find_char(231))
# convert_img_from_path("Pilot Wallpaper.png", scale=0.04)
convert_vid_from_path("spinning_dodecahedron.mp4", scale=0.01)

end = time.time()
print(f"Total Elapsed: {end-start} seconds")
