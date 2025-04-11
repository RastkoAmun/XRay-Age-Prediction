import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

image_dir = Path("../data/processed/training-set")
labels = pd.read_csv("../submission.csv")
labels["error"] = abs(labels["real"] - labels["prediction"])

df_sorted = labels.sort_values(by="error", ascending=False)

fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    
# Show top 5 best predictions
for idx, row in enumerate(df_sorted.tail(5).itertuples()):
	img_path = image_dir / f"{int(row.id)}.png"
	image = Image.open(img_path).convert("L")

	axes[idx].imshow(image, cmap='gray')
	axes[idx].axis("off")
	axes[idx].set_title(f"ID:{row.id}\nReal: {int(row.real)}\nPred: {int(row.prediction)}")

plt.suptitle("Top 5 Best Predictions", y=1.05)
plt.tight_layout()
plt.show()
plt.close(fig)
