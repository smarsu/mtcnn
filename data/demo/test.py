import cv2
import numpy as np

raw_img = cv2.imread('bigxx.jpg')
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

def integral(img, mean, std):
    img = img.astype(np.float32)
    img -= mean
    img /= std
    h, w = img.shape

    for idx in range(1, h):
        img[idx:idx+1] += img[idx-1:idx]
    for idx in range(1, w):
        img[:, idx:idx+1] += img[:, idx-1:idx]

    zeros = np.zeros(shape=[h + 1, w + 1])
    zeros[1:h+1, 1:w+1] = img
    img = zeros
    return img


def harr(img, mean, std):
    sum_img = integral(img, mean, std)
    print(sum_img[:9][:9])
    p1 = (sum_img[7][3] - sum_img[2][3]) / 15 * std
    p2 = (sum_img[7][6] - sum_img[7][3] + sum_img[2][3] - sum_img[2][6]) / 15 * std * -2
    p3 = (sum_img[7][9] - sum_img[7][6] + sum_img[2][6] - sum_img[2][9]) / 15 * std

    p1 = (sum_img[7][3] - sum_img[2][3]) / 15 * std
    p2 = (sum_img[7][6] - sum_img[7][3] + sum_img[2][3] - sum_img[2][6]) / 15 * std * -2
    p3 = (sum_img[7][9] - sum_img[7][6] + sum_img[2][6] - sum_img[2][9]) / 15 * std
    return p1 + p2 + p3

if __name__ == '__main__':
    det64 = harr(raw_img, 0, 1)
    det16 = harr(raw_img, 128, 512)
    print(det64)
    print(det16)
