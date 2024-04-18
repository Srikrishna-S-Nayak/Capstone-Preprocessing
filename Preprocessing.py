import cv2


def gaussian_mixture_model(video_path):
    vid = cv2.VideoCapture(video_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=15)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        back_sub_mask = back_sub.apply(frame)
        back_sub_mask = cv2.morphologyEx(back_sub_mask, cv2.MORPH_OPEN, kernel)

        frame2 = cv2.GaussianBlur(frame, (3, 3), 0)
        back_sub_mask2 = back_sub.apply(frame2)
        back_sub_mask2 = cv2.morphologyEx(back_sub_mask2, cv2.MORPH_OPEN, kernel)
        concatenated_frames = cv2.hconcat([back_sub_mask, back_sub_mask2])
        cv2.imshow('Foreground Mask Vs FM with GB', concatenated_frames)

        # cv2.imshow('Foreground Mask', back_sub_mask)
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

        frame = cv2.resize(frame, (640, 360))

        # Enhance the frame
        denoised_vid = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        contrast_adjusted = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        contrast_adjusted_color = cv2.cvtColor(contrast_adjusted, cv2.COLOR_GRAY2BGR)

        concatenated_frames = cv2.hconcat([frame, contrast_adjusted_color])
        cv2.imshow('Original vs Enhanced', concatenated_frames)
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
