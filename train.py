import os 
import sys 
import numpy as np
import pandas as pd
from random import shuffle


def main():
    seed = 42

    np.random.seed(seed)
    
    current_dir = os.path.dirname(__file__)
    # add the keras_text_to_image module to the system path
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = current_dir + '\img_align_celeba'
    txt_dir_path = pd.read_csv('image_desc.csv')
    model_dir_path = current_dir

    img_width = 128
    img_height = 128
    img_channels = 3
    
    from DCGAN import DCGanV3 
    from utility.img_cap_loader import load_normalized_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)
    print(len(image_label_pairs))
    # print('image pairs',image_label_pairs)

    shuffle(image_label_pairs)

    gan = DCGanV3()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 200
    gan.glove_source_dir_path = './very_large_data'

    batch_size = 128
    epochs = 1000
    gan.fit(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
            snapshot_dir_path=current_dir + '/data/snapshots',
            snapshot_interval=100,
            batch_size=batch_size,
            epochs=epochs)


if __name__ == '__main__':
    main()