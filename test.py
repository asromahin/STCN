import torch
import cv2
import matplotlib.pyplot as plt

import os
import shutil
import numpy as np
from tqdm import tqdm
import glob
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class InferPlot:
    def __init__(self, first_frame):
        self.fig, self.axs = plt.subplots(figsize=(16, 16))

        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.points = []
        self.image = first_frame
        self.mask = None
        self.show_im = self.axs.imshow(self.image)
        plt.tight_layout()
        plt.show()

    def onclick(self, event):
        # start = time.time()
        # print(event)
        self.points.append((event.xdata, event.ydata))
        points = np.array(self.points)
        points = points.astype(int)

        plot_image = self.image.copy()
        self.mask = np.zeros((plot_image.shape[0], plot_image.shape[1]), dtype='uint8')
        cv2.drawContours(plot_image, [points], -1, (0, 255, 0), 1)
        cv2.drawContours(self.mask, [points], -1, 255, -1)
        self.show_im.set_data(plot_image)
        self.fig.canvas.draw()
        # print(time.time() - start)


TEMP_FOLDER = 'temp'
IMAGES_PATH = os.path.join(TEMP_FOLDER, 'JPEGImages')
ANNOT_PATH = os.path.join(TEMP_FOLDER, 'Annotations')
MAX_FRAMES = 500

if __name__ == '__main__':
    shutil.rmtree(TEMP_FOLDER, ignore_errors=True)

    # video_path = 'data/Chickens_Outside_04_1.mp4'
    video_path = r'C:\datasets\tmrw\tenor\acewink-chatroyale.gif'
    video_name = os.path.split(video_path)[-1].split('.')[0]

    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    inf = InferPlot(first_frame)

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.join(IMAGES_PATH, video_name), exist_ok=True,)
    os.makedirs(os.path.join(ANNOT_PATH, video_name), exist_ok=True,)

    idx = 0
    cv2.imwrite(os.path.join(IMAGES_PATH, video_name,  str(idx).zfill(5)+'.jpg'), first_frame)
    cv2.imwrite(os.path.join(ANNOT_PATH, video_name, str(idx).zfill(5)+'.png'), inf.mask)

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(IMAGES_PATH, video_name, str(idx).zfill(5)+'.jpg'), frame)
        idx += 1
        if idx > MAX_FRAMES - 1:
            break

    video_annot_path = os.path.join(TEMP_FOLDER, 'Annotations')
    os.system(f"python eval_generic.py --data_path {TEMP_FOLDER} --output {video_annot_path}")\

    images = sorted(glob.glob(os.path.join(IMAGES_PATH, '**', '*')))
    masks = sorted(glob.glob(os.path.join(ANNOT_PATH, '**', '*')))

    im = cv2.imread(images[0])
    writer = cv2.VideoWriter(os.path.join(TEMP_FOLDER, 'video.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (im.shape[1], im.shape[0]*2))
    for i in tqdm(range(len(images))):
        im = cv2.imread(images[i])
        mask = cv2.imread(masks[i])

        cat_im = np.concatenate([im, mask], axis=0)
        writer.write(cat_im)
    writer.release()


