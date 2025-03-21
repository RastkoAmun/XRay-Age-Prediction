import pandas as pd
from PIL import Image
from pathlib import Path
import sys

set_type = sys.argv[1]

ROOT_PATH = Path(f'../data/boneage-{set_type}-dataset')
SAVE_PATH = Path(f'../data/processed/{set_type}-set')

# load labels, get only ids and transform it numpy array
labels = pd.read_csv(f'../data/boneage-{set_type}-dataset.csv')
image_ids = labels['id'] if sys.argv[1] == 'training' else labels['Case ID']
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