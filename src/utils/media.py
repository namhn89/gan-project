import cv2
import imageio
import glob
import os
import natsort
import re
import PIL


def make_gif(folder, pattern="*.png", file_path='./out.gif'):
    images = []
    for filename in sorted(glob.glob(os.path.join(folder, pattern)),
                           key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]):
        # print(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave(file_path, images)


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


def create_dif(folder, pattern, output_path="out.gif"):
    with imageio.get_writer(output_path, mode="I") as writer:
        filenames = glob.glob(folder + pattern)
        filenames = sorted(filenames, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def getFrame(video_capture, sec, count, save_dir):
    video_capture.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = video_capture.read()
    if hasFrames:
        cv2.imwrite(f"{save_dir}/image_{count:04d}.jpg", image)  # save frame as JPG file
    return hasFrames


def video_to_frames(video_path, save_dir="./", fps=24):
    video_capture = cv2.VideoCapture(video_path)

    sec = 0
    frameRate = 1 / fps  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(video_capture, sec, count, save_dir)

    while success:
        print(count)
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(video_capture, sec, count, save_dir)


def frames_to_video(frames, video_path, fps=24):
    height, width = frames[0].shape[:2]
    size = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        # writing to an image array
        out.write(frames[i].astype('uint8'))
    out.release()


def frame_files_to_video(frame_files=None, video_path=None, fps=24):
    frame_array = []
    for filename in frame_files:
        # reading each files
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def load_image(image_path, color_space="rgb"):
    image = cv2.imread(image_path)

    assert image is not None

    if color_space == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError

    return image
