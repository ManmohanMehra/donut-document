from inference import run_inference
import os

# Let's find one image from your uploaded data to test
test_image_dir = 'data/augmented/'
test_images = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if test_images:
    sample_path = os.path.join(test_image_dir, test_images[0])
    print(f"Testing model on: {sample_path}")
    
    result = run_inference(sample_path)
    
    print("\n--- Extracted Data ---")
    import json
    print(json.dumps(result, indent=2))
else:
    print("No images found in data/augmented/ to test with.")