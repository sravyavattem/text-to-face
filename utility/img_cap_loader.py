import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width, img_height):


    images = dict()
    texts = dict()
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.jpg'):
            # name = f.replace('.jpg', '.jpg')
            images[f] = filepath
    # for f in os.listdir(txt_dir_path):
    #     filepath = os.path.join(txt_dir_path, f)
    #     if os.path.isfile(filepath) and f.endswith('.txt'):
    #         name = f.replace('.txt', '')
    #         texts[name] = open(filepath, 'rt').read()
    
    image_names = list(txt_dir_path['images'])
    text = list(txt_dir_path['annots'])
    for i,j in zip(image_names,text):
        texts[i] = j
    
    # print(images)
    # print(texts)

    result = []
    for name, img_path in images.items():
        if name in texts:
            text = texts[name]
            image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
            image = (image.astype(np.float32) / 255)
            result.append([image, text])
    # print(len(result))
    return np.array(result[0:1000])