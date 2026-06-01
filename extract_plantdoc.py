import zipfile
import os
import re

def sanitize_filename(filename):
    # Characters not allowed in Windows: < > : " / \ | ? *
    # We keep / for directory structure
    path_parts = filename.split('/')
    sanitized_parts = []
    for part in path_parts:
        # Replace ? with _
        sanitized = part.replace('?', '_')
        # Replace other invalid chars with _
        sanitized = re.sub(r'[<>:"\\|*]', '_', sanitized)
        sanitized_parts.append(sanitized)
    return '/'.join(sanitized_parts)

zip_path = 'plantdoc.zip'
extract_path = 'data_plantdoc'

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.infolist():
        sanitized_name = sanitize_filename(member.filename)
        # Remove the top-level folder name from the zip (PlantDoc-Dataset-master/)
        target_path = os.path.join(extract_path, *sanitized_name.split('/')[1:])
        
        if member.is_dir():
            if not os.path.exists(target_path):
                os.makedirs(target_path)
        else:
            # Ensure parent directory exists
            parent = os.path.dirname(target_path)
            if not os.path.exists(parent):
                os.makedirs(parent)
            
            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())

print(f"Extraction complete to {extract_path}")
