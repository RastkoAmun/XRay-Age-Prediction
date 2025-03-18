import pandas as pd
from PIL import Image
from pathlib import Path

ROOT_PATH = Path('../data/boneage-training-dataset')
SAVE_PATH = Path('../data/processed/training-set')

# load labels, get only ids and transform it numpy array
training_labels = pd.read_csv('../data/boneage-training-dataset.csv')
image_ids = training_labels['id']
image_ids = image_ids.to_numpy()


# loop through all ids and use them to load images one by one
for img_id in image_ids:
  # Loads image path, opens image with PIL library and resizes it to (256, 344) resolution
  img_path = ROOT_PATH / str(img_id)
  img_path = img_path.with_suffix('.png')
  img = Image.open(img_path).resize((256, 344))

  # Makes save folder and saves each newly resized image
  save_folder = SAVE_PATH
  save_folder.mkdir(parents=True, exist_ok=True)
  img.save((save_folder / f"{img_id}.png"))

# Takes around 4 minutes to run on Apple's M2 PRO chip
print("Image resizing finished!")