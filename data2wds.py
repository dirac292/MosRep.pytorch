# import os
# import os.path as osp
# import random
# import argparse
# import webdataset as wds
# from PIL import Image

# parser = argparse.ArgumentParser(description='Webdataset conversion for unlabeled images')
# parser.add_argument('-i', '--data-dir', metavar='DIR', help='path to image directory')
# parser.add_argument('-o', '--output-dir', metavar='DIR', help='path to output folder')

# def readfile(fname):
#     with open(fname, "rb") as stream:
#         return stream.read()

# def wds_shards_create(args):
#     # Get list of all images in directory
#     image_files = []
#     for root, _, files in os.walk(args.data_dir):
#         for file in files:
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_files.append(os.path.join(root, file))
    
#     print(f"Found {len(image_files)} images.")
    
#     os.makedirs(osp.join(args.output_dir, 'train'), exist_ok=True)
#     print('Initializing dataset creation...')

#     train_keys = set()
#     indexes = list(range(len(image_files)))
#     random.shuffle(indexes)

#     with wds.ShardWriter(osp.join(args.output_dir, 'train', 'shards-%05d.tar'), maxcount=1000) as sink:
#         for i in indexes:
#             fname = image_files[i]
#             try:
#                 # Verify the image can be opened and convert if necessary
#                 with Image.open(fname) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
                
#                 # Read image data
#                 image = readfile(fname)
#                 key = os.path.splitext(os.path.basename(fname))[0]

#                 # Ensure unique keys
#                 if key in train_keys:
#                     print(f"Duplicate key detected and skipped: {key}")
#                     continue
#                 train_keys.add(key)

#                 # Create sample with required keys
#                 sample = {
#                     "__key__": str(i),            # Unique key for each sample
#                     "fname": osp.basename(fname),  # Filename for reference
#                     "data": image,                # Image data under 'data' key
#                     "jpg": image                  # Compatibility key
#                 }

#                 sink.write(sample)
#             except Exception as e:
#                 print(f"Error processing {fname}: {str(e)}")
#                 continue
    
#     print('Dataset creation finished.')

# if __name__ == '__main__':
#     args = parser.parse_args()
#     wds_shards_create(args)

import os
import os.path as osp
import random
import argparse
import webdataset as wds

parser = argparse.ArgumentParser(description='WebDataset conversion')
parser.add_argument('-i', '--data-dir', metavar='DIR', help='path to your dataset directory')
parser.add_argument('-o', '--output-dir', metavar='DIR', help='path to output folder')

def readfile(fname):
    with open(fname, "rb") as stream:
        return stream.read()

def get_image_paths(data_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def wds_shards_create(args):
    # Get all image paths
    image_paths = get_image_paths(args.data_dir)
    print(f"Found {len(image_paths)} images in {args.data_dir}")
    random.shuffle(image_paths)

    # Create output directory
    os.makedirs(osp.join(args.output_dir, 'train'), exist_ok=True)
    print('Initialized dataset directory')

    # Process training set
    train_keys = set()
    with wds.ShardWriter(osp.join(args.output_dir, 'train', 'shards-%05d.tar'), maxcount=1000) as sink:
        for idx, fname in enumerate(image_paths):
            image = readfile(fname)
            key = os.path.splitext(os.path.basename(fname))[0]
            if key in train_keys:
                print(f"Duplicate key {key}, skipping.")
                continue
            train_keys.add(key)

            cls = -1  # Since images are unlabeled, set cls to -1
            sample = {"__key__": str(idx), "jpg": image, "cls": cls}
            sink.write(sample)
    print('Finished processing training set')

if __name__ == '__main__':
    args = parser.parse_args()
    wds_shards_create(args)

