from PIL import Image
import os
import cv2 
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm
import concurrent.futures as CONC

# ------------- 1. Chargement des documents -------------
def is_valide(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def images_checking(data_path,images_extensions,pdf_extensions):
    valide_images = []
    for file_name in os.listdir(data_path):
        image_path = os.path.join(data_path,file_name)
        if is_valide(image_path):
            if os.path.splitext(file_name)[1].lower() in pdf_extensions:
                image_path_pdf = convert_from_path(Image.open(image_path))
                valide_images.append(image_path_pdf)
            elif os.path.splitext(file_name)[1].lower() in images_extensions:
                valide_images.append(image_path)
            else:
                print(f'Image {image_path}, have invalid extension')
    return valide_images

# ------------- 2. Prétraitement d’image -------------
def binarizing(image_path):
    numerical_image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(numerical_image,cv2.COLOR_BGR2GRAY)
    _,binarized = cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY)
    return binarized

def parallel_binarizing(valide_images):
    with tqdm(total=len(valide_images), desc="Binarizing ...", ncols=120) as pbar:
        def process_one(image_path):
            result = binarizing(image_path)
            pbar.update(1)
            return result
        with CONC.ThreadPoolExecutor(max_workers=4) as executor:
            binarized_images = list(executor.map(process_one, valide_images))
    print(f'We got {len(binarized_images)} Binarized images.')
    return binarized_images

"""
if show:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(denoised)
    ax[1].set_title('Image with Median Filter')
    ax[1].axis('off')
    plt.show()
    show = False
    
"""

def parallel_denoising(binarized_images):
    with tqdm(total=len(binarized_images), desc="Denoising ....",colour='red', ncols=120) as pbar:
        def process_one(image_path):
            result = median_filter(np.array(image_path),3)
            pbar.update(1)
            return result
        with CONC.ThreadPoolExecutor(max_workers=4) as executor:
            denoised_images = list(executor.map(process_one, binarized_images))
    print(f'We got {len(denoised_images)} Denoised images.')
    return denoised_images

# ------------- MAIN -------------
data_path = 'batch_1'
images_extensions = ['.png','.jpg','.tiff']
pdf_extensions = ['.pdf']

valide_images = images_checking(data_path,images_extensions,pdf_extensions)
print(f'We Have {len(valide_images)} Images in our {data_path} folder')
binarized_images = parallel_binarizing(valide_images)
denoised_images = parallel_denoising(binarized_images)