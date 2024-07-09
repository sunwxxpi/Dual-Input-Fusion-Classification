import os
import shutil
import pandas as pd
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

# Define augmentation sequence with more random elements
seq_aug = iaa.Sequential([
    iaa.Fliplr(0.5),                             # 50% 확률의 좌우 반전
    iaa.Crop(percent=(0, 0.1)),                  # 0 ~ 10% 사이의 무작위로 Crop
    iaa.GaussianBlur(sigma=(0, 1)),              # 0 ~ 1.0 사이의 무작위 Blur
    # Add more augmentations as needed
])

# Function to augment images and masks
def augment_images_and_masks(imgs, masks, num_aug_images):
    augmented_images = []
    augmented_masks = []

    for _ in range(num_aug_images):
        det_seq_aug = seq_aug.to_deterministic()  # Ensure the same augmentation is applied to all images
        
        augmented_images.append([det_seq_aug(image=img) for img in imgs])
        augmented_masks.append([det_seq_aug(image=mask) for mask in masks])

    return augmented_images, augmented_masks

# Function to normalize image values to the range [0, 1]
def normalize_image(img):
    if img.max() > 1.0:
        img = img / 255.0
        
    return img

# Function to copy and augment images in directories
def copy_and_augment_images_in_directories(labels, directories, save_directories, target_count_per_class):
    # Create save directories if they don't exist
    for save_directory in save_directories:
        os.makedirs(save_directory, exist_ok=True)

    # Copy original images to the save directories
    for img_fn in labels['name']:
        for directory, save_directory in zip(directories, save_directories):
            src_path = os.path.join(directory, img_fn)
            dst_path = os.path.join(save_directory, img_fn)
            shutil.copy(src_path, dst_path)
    
    # Augment images to balance classes
    new_rows = []
    for label in labels['label'].unique():
        class_imgs = labels[labels['label'] == label]
        current_count = len(class_imgs)
        num_aug_needed = target_count_per_class - current_count
        
        if num_aug_needed > 0:
            num_aug_per_image = (num_aug_needed // current_count) + 1

            for idx, row in class_imgs.iterrows():
                img_fn = row['name']
                imgs = [mpimg.imread(os.path.join(directories[i], img_fn)) for i in range(2)]  # Normal images
                masks = [mpimg.imread(os.path.join(directories[2], img_fn))]  # Mask images

                augmented_images, augmented_masks = augment_images_and_masks(imgs, masks, num_aug_per_image)

                for i, (aug_set, aug_mask_set) in enumerate(zip(augmented_images, augmented_masks)):
                    if current_count >= target_count_per_class:
                        break

                    for img_aug, save_directory in zip(aug_set, save_directories[:2]):
                        base_fn, ext = os.path.splitext(img_fn)
                        img_aug_fn = os.path.join(save_directory, f"{base_fn}_aug{i + 1}{ext}")
                        
                        # Normalize image before saving
                        img_aug = normalize_image(img_aug)
                        mpimg.imsave(img_aug_fn, img_aug)

                    for mask_aug, save_directory in zip(aug_mask_set, save_directories[2:]):
                        base_fn, ext = os.path.splitext(img_fn)
                        mask_aug_fn = os.path.join(save_directory, f"{base_fn}_aug{i + 1}{ext}")
                        mpimg.imsave(mask_aug_fn, mask_aug, cmap='gray')  # Save mask without normalization

                    # Save augmented image label
                    new_rows.append({'name': f"{base_fn}_aug{i + 1}{ext}", 'label': label})
                    current_count += 1

    new_labels = pd.concat([labels, pd.DataFrame(new_rows)], ignore_index=True)
    
    return new_labels

# Load label.csv
label_path = './data/KO/train/label.csv'
labels = pd.read_csv(label_path)

# Calculate target count per class (3 times the total number of classes, distributed equally)
total_target_count = len(labels) * 3
num_classes = len(labels['label'].unique())
target_count_per_class = total_target_count // num_classes

# Define directories
img_directory = './data/KO/train/img'
elastogram_directory = './data/KO/train/elastogram'
mask_directory = './data/KO/train/mask'

save_img_directory = './data/KO/train/img_aug'
save_elastogram_directory = './data/KO/train/elastogram_aug'
save_mask_directory = './data/KO/train/mask_aug'

# Augment images in the directories, balance classes, and save to another directory
directories = [img_directory, elastogram_directory, mask_directory]
save_directories = [save_img_directory, save_elastogram_directory, save_mask_directory]
augmented_labels = copy_and_augment_images_in_directories(labels, directories, save_directories, target_count_per_class)

# Save the updated labels to a new CSV
augmented_labels.to_csv('./data/KO/train/label_aug.csv', index=False)