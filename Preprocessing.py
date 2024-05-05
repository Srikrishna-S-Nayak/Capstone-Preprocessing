import cv2
import numpy as np


def gaussian_mixture_model(video_path):
    vid = cv2.VideoCapture(video_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=15)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        back_sub_mask = back_sub.apply(frame)
        back_sub_mask = cv2.morphologyEx(back_sub_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask with Gaussian Blurring', back_sub_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def enhance_video(video_path):
    vid = cv2.VideoCapture(video_path)
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Could not read video at frame {}".format(frame))
            break

        frame = cv2.resize(frame, (960, 540))
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

        b, g, r = cv2.split(frame)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        contrast_adjusted_color = cv2.merge((b, g, r))

        concatenated_frames = cv2.hconcat([frame, contrast_adjusted_color])
        cv2.imshow('Original vs Enhanced', concatenated_frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def test(video_path):
    vid = cv2.VideoCapture(video_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=15)
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Could not read video at frame {}".format(frame))
            break

        frame = cv2.resize(frame, (960, 540))
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

        b, g, r = cv2.split(frame)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        contrast_adjusted_color = cv2.merge((b, g, r))

        contrast_adjusted_color = cv2.GaussianBlur(contrast_adjusted_color, (3, 3), 0)
        back_sub_mask = back_sub.apply(contrast_adjusted_color)
        contrast_adjusted_mask = cv2.morphologyEx(back_sub_mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask with Gaussian Blurring', contrast_adjusted_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    vid_path = "./data/D19_20230511115642.mp4"

    choice = input("1 for Gaussian Mixture Model\n2 for Enhance Video \n")
    if choice == "1":
        gaussian_mixture_model(vid_path)
    elif choice == "2":
        enhance_video(vid_path)
    elif choice == "3":
        test(vid_path)
