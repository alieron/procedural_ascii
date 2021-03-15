import math
# python -m pip install opencv-python
import cv2
import numpy as np                                  # python -m pip install numpy
import time
from PIL import Image, ImageDraw, ImageFont


ascii_order = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


def find_char(brightness):
    return ascii_order[math.floor(brightness * (len(ascii_order) / 256))]


def img_to_ascii(img, fontname, scale, char_width, char_height, fontsize):
    w, h = img.size
    img = img.resize(
        (int(scale * w), int(scale * h * (char_width / char_height))), Image.NEAREST)
    w, h = img.size

    raw_pix = img.load()

    # font = ImageFont.truetype(f"fonts\\{fontname}.ttf", fontsize)                         # Numba Error
    font = ImageFont.truetype("fonts\\%s.ttf" % fontname, fontsize)

    output_img = Image.new("RGB", (w * char_width, h * char_height), (0, 0, 0))
    d = ImageDraw.Draw(output_img)

    for i in range(h):
        for j in range(w):
            color = raw_pix[j, i]
            brightness = sum(color) / 3
            d.text((j * char_width, i * char_height),
                   find_char(brightness), font=font, fill=color)

    return output_img


def convert_img_from_path(image_path, output_file="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    img = Image.open(image_path).convert("RGB")
    output_img = img_to_ascii(img, fontname, scale,
                              char_width, char_height, fontsize)
    # output_img.save(f"{output_file}.png")                                                                 # Numba Error
    output_img.save("%s.png" % output_file)

    # print(f"Image {output_file}.png has been successfully generated from {image_path}.")                  # Numba Error
    print("Image %s.png has been successfully generated from %s" %
          (output_file, image_path))


def vid_to_ascii(frame_array, fontname, scale, char_width, char_height, fontsize):
    output_frame_array = []

    t5 = time.time()
    for f in frame_array:
        img = Image.fromarray(f)
        output_frame_array.append(img_to_ascii(
            img, fontname, scale, char_width, char_height, fontsize))
    t6 = time.time()

    print(f"Images to ASCII took {t6-t5} seconds")

    return output_frame_array


def convert_vid_from_path(video_path, output_file="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    vid = cv2.VideoCapture(video_path)

    frame_array = []

    t1 = time.time()
    # TODO: BGR2RGB potential speed up using numpy & numba
    for f in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        _, img = vid.read()
        frame_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t2 = time.time()

    output_frame_array = vid_to_ascii(
        frame_array, fontname, scale, char_width, char_height, fontsize)

    w, h = output_frame_array[0].size
    fps = vid.get(cv2.CAP_PROP_FPS)
    # output_vid = cv2.VideoWriter(f"{output_file}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))      # Numba Error
    output_vid = cv2.VideoWriter(
        "%s.mp4" % output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    t3 = time.time()
    for i in range(len(output_frame_array)):
        output_vid.write(cv2.cvtColor(
            np.array(output_frame_array[i]), cv2.COLOR_RGB2BGR))
    t4 = time.time()

    # print(f"Video {output_file}.mp4 has been successfully generated from {video_path}.")                  # Numba Error
    print("Video %s.mp4 has been successfully generated from %s" %
          (output_file, video_path))
    print(f"BGR2RGB took {t2-t1} seconds")
    print(f"RGB2BGR took {t4-t3} seconds")


def convert_vid_from_dir(dir, output_dir="output", fontname="JetBrainsMono-Regular", scale=0.08, char_width=10, char_height=20, fontsize=15):
    pass


start = time.time()

# print(find_char(231))
convert_img_from_path("Pilot Wallpaper.png", scale=0.04)
# convert_vid_from_path("spinning_dodecahedron.mp4", scale=0.02)

end = time.time()
print(f"Total Elapsed: {end-start} seconds")
