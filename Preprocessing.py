import cv2
import numpy as np

x1 = 0
y1 = 0
x2 = 0
y2 = 0
x3 = 0
y3 = 0
x4 = 0
y4 = 0
x21 = 100
y21 = 100
x22 = 100
y22 = 100
x23 = 100
y23 = 100
x24 = 100
y24 = 100

src_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # Specify source points
dst_points = np.float32([[x21, y21], [x22, y22], [x23, y23], [x24, y24]])  # Specify destination points


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
        cv2.imshow('Foreground Mask with Guassian Blurring', back_sub_mask)
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


def perspective_transform(video_path, src_points, dst_points):
    vid = cv2.VideoCapture(video_path)

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Could not read video at frame {}".format(frame))
            break
        frame = cv2.resize(frame, (640, 360))
        # Compute perspective transform matrix
        m = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply perspective transformation
        transformed = cv2.warpPerspective(frame, m, (frame.shape[1], frame.shape[0]))

        cv2.imshow('Original video', frame)
        cv2.imshow('Perspective Transformed', transformed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    vid_path = "./data/D19_20230511115642.mp4"

    choice = input("1 for Gaussian Mixture Model\n2 for Enhance Video \n3 for Perspective Transform\n")
    if choice == "1":
        gaussian_mixture_model(vid_path)
    elif choice == "2":
        enhance_video(vid_path)
    elif choice == "3":
        perspective_transform(vid_path, src_points, dst_points)
