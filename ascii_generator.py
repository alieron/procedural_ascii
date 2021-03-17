import math
import cv2                                      # python -m pip install opencv-python
import numpy as np                              # python -m pip install numpy
from time import time
from tqdm import trange                         # python -m pip install tqdm
from PIL import Image, ImageDraw, ImageFont


ascii_order = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


def find_char(gray):
    return ascii_order[math.floor(gray * (len(ascii_order) / 256))]


# class create_font:
#     def __init__(self, fontname, fontsize=15):
#         self.fontname = fontname
#         self.fontsize=fontsize
#     # make the character order from json


class ASCII_Generator:
    def __init__(self, mode="color"):
        # This is where we want to start working with fonttools
        self.font = ImageFont.truetype("fonts\\JetBrainsMono-Regular.ttf", 15)
        self.mode = mode
        self.char_width = 10
        self.char_height = 20

    def _img_rgb_to_ascii(self, img_rgb, w, h, scale):
        img_rgb = img_rgb.resize((w, h), Image.ANTIALIAS)

        img_out = Image.new("RGB", (w * self.char_width,
                                    h * self.char_height), (0, 0, 0))
        d = ImageDraw.Draw(img_out)

        pix_data = img_rgb.load()

        for i in range(h):
            for j in range(w):
                rgb = pix_data[j, i]
                gray = sum(rgb) / 3
                d.text((j * self.char_width, i * self.char_height),
                       find_char(gray), font=self.font, fill=rgb)

        return img_out

    def _vid_to_ascii(self, frame_array, scale):

        output_frame_array = []

        w, h = frame_array[0].size
        w, h = int(scale * w), int(scale * h *
                                   (self.char_width / self.char_height))

        # for f in frame_array:
        #     f = Image.fromarray(f)
        #     output_frame_array.append(img_rgb_to_ascii(
        #         f, fontname, scale, char_width, char_height, fontsize))

        for k in trange(len(frame_array)):
            img_out = self._img_rgb_to_ascii(frame_array[k], w, h, scale)
            output_frame_array.append(img_out)

        # w, h = w*self.char_width, h*self.char_height
        w, h = output_frame_array[0].size

        return output_frame_array, w, h
        # return output_frame_array

    def convert_img(self, image_path, output_file="output.png", scale=0.08):
        print("Converting Image...")

        t1 = time()

        img_rgb = Image.open(image_path).convert("RGB")

        w, h = img_rgb.size
        w, h = (int(scale * w), int(scale * h *
                                    (self.char_width / self.char_height)))

        img_out = self._img_rgb_to_ascii(img_rgb, w, h, scale)
        img_out.save(output_file)

        t2 = time()
        print("\nImage %s has been successfully generated from %s, completed in %g seconds" %
              (output_file, image_path, round(t2-t1, 2)))

    def convert_vid(self, video_path, output_file="output.mp4", scale=0.05):
        print("Converting Video...")
        t1 = time()

        vid = cv2.VideoCapture(video_path)

        frame_array = []

        # Can't use multiprocessing, or at least I don't think you can
        print("\nSplitting video into its individual frames...")
        for f in trange(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img_bgr = vid.read()
            # try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = Image.fromarray(img_rgb)
            frame_array.append(img_rgb)
            # except:
            #     break

        print("\nConverting each frame to ASCII.")
        output_frame_array, w, h = self._vid_to_ascii(frame_array, scale)
        # output_frame_array = self._vid_to_ascii(frame_array, scale)
        # w, h = output_frame_array[0].size

        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_vid = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

        print("\nWriting frames to new file.")
        for i in range(len(output_frame_array)):
            # try:
            img_bgr = np.array(output_frame_array[i])
            output_vid.write(cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR))
            # except:
            #     output_vid.release()
            #     break

        cv2.destroyAllWindows()
        output_vid.release()

        t2 = time()
        print("Video %s has been successfully generated from %s, completed in %g seconds." %
              (output_file, video_path, round(t2-t1, 2)))

    def convert_vid_to_dir(self, dir, output_dir="output", scale=0.05):
        pass
