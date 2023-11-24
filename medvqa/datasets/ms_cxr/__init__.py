import random
import os
import pandas as pd
from dotenv import load_dotenv
from zipfile import ZipFile
from medvqa.datasets.segmentation_utils import compute_mask_from_bounding_boxes
from medvqa.datasets.mimiccxr import MIMICCXR_JPG_IMAGES_LARGE_DIR

load_dotenv()

MS_CXR_LOCAL_ALIGNMENT_CSV_PATH = os.environ['MS_CXR_LOCAL_ALIGNMENT_V1.0.0_CSV_PATH']

class PhraseGroundingAnnotationsVisualizer:

    _COLORS = [
        (1.0, 0.0, 0.0), # red
        (0.0, 0.0, 1.0), # blue
        (1.0, 0.0, 1.0), # magenta
        (0.0, 1.0, 1.0), # cyan
        (0.0, 1.0, 0.0), # green
    ]

    def __init__(self):
        self.df = pd.read_csv(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH)
        self.dicom_id_2_rows = {}
        for _, row in self.df.iterrows():
            dicom_id = row['dicom_id']
            if dicom_id not in self.dicom_id_2_rows:
                self.dicom_id_2_rows[dicom_id] = []
            self.dicom_id_2_rows[dicom_id].append(row)

    def visualize_row(self, row_idx):
        assert 0 <= row_idx < len(self.df)
        row = self.df.iloc[row_idx]
        image_path = os.path.join(MIMICCXR_JPG_IMAGES_LARGE_DIR, row['path'][6:]) # remove 'files/'
        label_text = row['label_text']
        category_name = row['category_name']
        x = row['x']
        y = row['y']
        w = row['w']
        h = row['h']
        print(f'Image path: {image_path}')
        print(f'Label text: {label_text}')
        print(f'Category name: {category_name}')
        print(f'x: {x}, y: {y}, w: {w}, h: {h}')
        # show image with bounding box
        from PIL import Image
        from matplotlib import pyplot as plt
        img = Image.open(image_path)
        img = img.convert('RGB')
        plt.imshow(img)
        plt.axis('off')
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        plt.show()

    def visualize_dicom_id(self, dicom_id, figsize=(10, 10)):
        assert dicom_id in self.dicom_id_2_rows
        rows = self.dicom_id_2_rows[dicom_id]
        print(f'Found {len(rows)} rows for dicom_id {dicom_id}')
        # show image with bounding box
        from PIL import Image
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=figsize)
        image_path = os.path.join(MIMICCXR_JPG_IMAGES_LARGE_DIR, rows[0]['path'][6:])
        img = Image.open(image_path)
        img = img.convert('RGB')
        plt.imshow(img)
        plt.axis('off')
        # Assign color to each text label
        label_text_2_color = {}
        for row in rows:
            label_text = row['label_text']
            if label_text not in label_text_2_color:
                label_text_2_color[label_text] = len(label_text_2_color)
        # Create a Rectangle patch for each row
        import matplotlib.patches as patches        
        for row in rows:
            x = row['x']
            y = row['y']
            w = row['w']
            h = row['h']
            print(f'Label text: {row["label_text"]}')
            print(f'Category name: {row["category_name"]}')
            print(f'x: {x}, y: {y}, w: {w}, h: {h}')
            color = self._COLORS[label_text_2_color[row['label_text']]]
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(rect)
            # Add label text
            label_text = row['label_text']
            plt.text(x, y-10, label_text, fontsize=12, color=color)
        plt.show()

def export_images_and_annotations_to_zip(folder_path, zip_filename, num_images=None):
    # Image paths
    df = pd.read_csv(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH)
    image_paths = []
    for _, row in df.iterrows():
        image_path = os.path.join(MIMICCXR_JPG_IMAGES_LARGE_DIR, row['path'][6:])  # remove 'files/'
        image_paths.append(image_path)
    
    # Remove duplicate image paths and sort them
    image_paths = list(set(image_paths))
    image_paths.sort()

    # Limit the number of images
    if num_images is not None:
        image_paths = random.sample(image_paths, num_images)

    # Create the output folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Create a zip file and add image files to it
    with ZipFile(os.path.join(folder_path, zip_filename), 'w') as zipf:
        for image_path in image_paths:
            assert os.path.exists(image_path) # Make sure the image file exists
            # Get the relative path within the zip
            relative_path = os.path.join('images', os.path.basename(image_path))
            zipf.write(image_path, relative_path)

        # Add the annotation CSV file to the zip
        zipf.write(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH, 'phrase_grounding_annotations.csv')

    print(f'Images and annotations exported to {zip_filename} in {folder_path}')

def get_ms_cxr_dicom_ids():
    df = pd.read_csv(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH)
    return list(set(df['dicom_id']))

def get_ms_cxr_dicom_id_2_phrases_and_masks(mask_height, mask_width):
    df = pd.read_csv(MS_CXR_LOCAL_ALIGNMENT_CSV_PATH)
    dicom_id_2_rows = {}
    for _, row in df.iterrows():
        dicom_id = row['dicom_id']
        if dicom_id not in dicom_id_2_rows:
            dicom_id_2_rows[dicom_id] = []
        dicom_id_2_rows[dicom_id].append(row)
    dicom_id_2_phrases_and_masks = {}
    for dicom_id, rows in dicom_id_2_rows.items():
        phrases = []
        masks = []
        phrase2bboxes = {}
        for row in rows:
            phrase = row['label_text']
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            iw, ih = row['image_width'], row['image_height']
            x /= iw
            y /= ih
            w /= iw
            h /= ih
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert 0 <= x + w <= 1
            assert 0 <= y + h <= 1
            if phrase not in phrase2bboxes:
                phrase2bboxes[phrase] = []
            phrase2bboxes[phrase].append((x, y, x + w, y + h))
        for phrase, bboxes in phrase2bboxes.items():
            mask = compute_mask_from_bounding_boxes(mask_height, mask_width, bboxes)
            phrases.append(phrase)
            masks.append(mask)
        dicom_id_2_phrases_and_masks[dicom_id] = (phrases, masks)
    return dicom_id_2_phrases_and_masks
