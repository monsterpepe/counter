import cv2
import hashlib
import numpy as np
import os
import requests

from img_urls import PUSHUP_UP, PUSHUP_DOWN

IMG_EXTENSION = '.jpg'
IMG_HEIGHT = 480
IMG_WIDTH = 480


def process_img(original_img, file_extension):
    original_arr = np.array(bytearray(original_img), dtype=np.uint8)
    cv2_img = cv2.imdecode(original_arr, -1)

    if cv2_img is not None:
        h, w = cv2_img.shape[:2]
        if h < w:
            cv2_img = cv2.resize(cv2_img, (IMG_WIDTH, int((IMG_WIDTH/w) * h)))
        else:
            cv2_img = cv2.resize(cv2_img, (int((IMG_HEIGHT/h) * w), IMG_HEIGHT))
        _, arr = cv2.imencode(file_extension, cv2_img)
        img = arr.tobytes()
        return img


def get_img(urls, img_path):
    for url in urls:
        try:
            r = requests.get(url)
            img = process_img(r.content, IMG_EXTENSION)
            if img is None:
                print(f'Img error: {url}')

            else:
                img_hash = hashlib.sha256(img).hexdigest()
                img_name = f'{img_hash}{IMG_EXTENSION}'

                if img_name in os.listdir(img_path):
                    print(f'Img hash exists (url: {url}, hash: {img_hash})')
                else:
                    with open(f'{img_path}/{img_name}', 'wb') as f:
                        f.write(img)
        except Exception as e:
            print(f'{e} (url: {url})')


def str_to_urls(url_str):
    urls = []
    for url in url_str.split('\n'):
        if url and url not in urls:
            urls.append(url)
    return urls


if __name__ == '__main__':
    paths = ('img', 'img/pushup_up', 'img/pushup_down')
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)
    urls = {
        'pushup_up': str_to_urls(PUSHUP_UP),
        'pushup_down': str_to_urls(PUSHUP_DOWN),
    }
    for pose_class, pose_urls in urls.items():
        get_img(pose_urls, f'img/{pose_class}')
