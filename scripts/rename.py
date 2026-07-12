import os
import glob

def rename_images_in_subdirs(main_dir):
    # Supported image extensions
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
    
    # Verify if the main directory exists
    if not os.path.exists(main_dir):
        print(f"Error: The directory '{main_dir}' does not exist.")
        return

    # Iterate through all items in the main directory
    for sub_dir in os.listdir(main_dir):
        sub_dir_path = os.path.join(main_dir, sub_dir)
        
        # Only process actual directories
        if os.path.isdir(sub_dir_path):
            
            # Determine the naming prefix based on your rules:
            # If the folder name is exactly 3 characters (e.g., IND, ARE), it's a PASSPORT.
            # Otherwise, use the folder name directly (e.g., QAT-RESIDENT-PERMIT).
            if len(sub_dir) == 3 and sub_dir.isalpha():
                prefix = f"{sub_dir}_PASSPORT"
            else:
                # Replaces hyphens with underscores if you prefer uniform naming (e.g., QAT_RESIDENT_PERMIT)
                prefix = sub_dir.replace('-', '_')
            
            # Find all images in the sub-directory
            images = []
            for ext in image_extensions:
                # case-insensitive matching by checking both upper and lower case
                images.extend(glob.glob(os.path.join(sub_dir_path, ext)))
                images.extend(glob.glob(os.path.join(sub_dir_path, ext.upper())))
            
            # Sort images to maintain a consistent ordering before renaming
            images.sort()
            
            if not images:
                print(f"No images found in: {sub_dir}")
                continue
                
            print(f"Processing folder '{sub_dir}' -> Prefix: '{prefix}' ({len(images)} images found)")
            
            # Rename the images sequentially
            for index, old_filepath in enumerate(images, start=1):
                # Extract the original file extension
                _, file_ext = os.path.splitext(old_filepath)
                
                # Format the new filename with 3-digit padding (001, 002, ... 00N)
                new_filename = f"{prefix}_{index:03d}{file_ext.lower()}"
                new_filepath = os.path.join(sub_dir_path, new_filename)
                
                try:
                    os.rename(old_filepath, new_filepath)
                except Exception as e:
                    print(f"  Failed to rename {os.path.basename(old_filepath)}: {e}")
                    
            print(f"Successfully processed '{sub_dir}'.\n")

if __name__ == "__main__":
    # Replace this with the actual path to your main data folder
    TARGET_DIRECTORY = "/Users/manu/Downloads/passport_data" 
    
    rename_images_in_subdirs(TARGET_DIRECTORY)