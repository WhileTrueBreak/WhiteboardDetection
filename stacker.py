import math
import os
from PIL import Image
import glob

# Get sorted list of all JPG images in the './output' folder
image_files = sorted(glob.glob("./output/*.jpg"))

if not image_files:
    print("No images found in ./output.")
    exit()

# Open and resize all images to 800x800
images = []
for file in image_files:
    img = Image.open(file)
    img_resized = img.resize((800, 800))
    images.append(img_resized)

# Set the resized dimensions
img_width, img_height = 800, 800

# Calculate grid dimensions (number of columns and rows)
cols = math.ceil(math.sqrt(len(images)))
rows = math.ceil(math.sqrt(len(images)))

# Create a new blank image to paste the grid images
combined_img = Image.new("RGB", (cols * img_width, rows * img_height))

# Paste each image into the correct position in the grid
for i, img in enumerate(images):
    x = (i % cols) * img_width
    y = (i // cols) * img_height
    combined_img.paste(img, (x, y))

# Save the combined image in the './output' folder
output_path = os.path.join("./output", "combined_grid.png")
combined_img.save(output_path)
print(f"Combined image saved to {output_path}")