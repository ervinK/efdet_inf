import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def inf(phi, weighted_bi, num_classes):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'



    model_path = 'efdet_model.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    
    score_threshold = 0.3
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

    

    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bi,
                            num_classes=num_classes,
                            score_threshold=score_threshold)
    model.load_weights(model_path, by_name=True)

    TEST_DIR = 'test_imgs/'

    test_img_list = os.listdir(TEST_DIR)
    for img_t in range(0 , len(test_img_list) * 10):
        image = cv2.imread(TEST_DIR + test_img_list[img_t%len(test_img_list)])
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        
        print(time.time() - start)

inf(0, False, 3)

