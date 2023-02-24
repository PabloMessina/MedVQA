import argparse
import os
import cv2
import multiprocessing as mp

def resize_images(src_dir, tgt_dir, target_size):
    # Find all image paths in source dir and nested subdirs
    # and resize them to target size in a matching folder in target dir    
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.jpg'):
                image_path = os.path.join(root, f)
                image = cv2.imread(image_path)
                # Resize image so that the smallest side is target_size
                h, w, _ = image.shape
                if h < w:
                    new_h = target_size
                    new_w = int(w * target_size / h)
                else:
                    new_w = target_size
                    new_h = int(h * target_size / w)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Save image to target dir
                tgt_image_path = image_path.replace(src_dir, tgt_dir)
                tgt_image_dir = os.path.dirname(tgt_image_path)
                os.makedirs(tgt_image_dir, exist_ok=True)
                cv2.imwrite(tgt_image_path, image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mimiccxr-images-source-dir', type=str, required=True)
    parser.add_argument('--mimiccxr-images-target-dir', type=str, required=True)
    parser.add_argument('--target-size', type=int, required=True)
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count())
    args = parser.parse_args()

    src_dir = args.mimiccxr_images_source_dir
    tgt_dir = args.mimiccxr_images_target_dir

    # Get part folders in source dir
    part_folders = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]    
    assert len(part_folders) == 10 # 10 parts in MIMICCXR dataset
    assert all(os.path.isdir(f) for f in part_folders)
    print('Found', len(part_folders), 'part folders in', src_dir)
    for part_folder in part_folders:
        print(part_folder)

    # Create target dir and part folders
    os.makedirs(tgt_dir, exist_ok=True)
    print('Created target dir', tgt_dir)
    for part_folder in part_folders:
        os.makedirs(part_folder.replace(src_dir, tgt_dir), exist_ok=True)
        print('Created target dir', part_folder.replace(src_dir, tgt_dir))

    # Resize images in parallel
    print('Resizing images in parallel using', args.num_workers, 'workers...')
    with mp.Pool(args.num_workers) as pool:
        pool.starmap(resize_images, [(part_folder, part_folder.replace(src_dir, tgt_dir), args.target_size)
            for part_folder in part_folders])

    print('Done')