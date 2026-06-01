import os
import shutil

src_train = 'data_plantdoc/train'
src_test = 'data_plantdoc/test'
dst = 'data_plantdoc_merged'

if not os.path.exists(dst):
    os.makedirs(dst)

def merge_folders(src, dst):
    for root, dirs, files in os.walk(src):
        relative_path = os.path.relpath(root, src)
        target_dir = os.path.join(dst, relative_path)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        for file in files:
            src_file = os.path.join(root, file)
            # Handle name collisions by prefixing if necessary
            dst_file = os.path.join(target_dir, file)
            if os.path.exists(dst_file):
                name, ext = os.path.splitext(file)
                dst_file = os.path.join(target_dir, f"{name}_coll{ext}")
            
            shutil.copy2(src_file, dst_file)

print("Merging train...")
merge_folders(src_train, dst)
print("Merging test...")
merge_folders(src_test, dst)
print("Merge complete.")
