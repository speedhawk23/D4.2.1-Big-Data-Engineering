import cv2
import matplotlib.pyplot as plt

image_paths = [
    r"D:\data\image_data\weather_image_recognition\rain\1532.jpg",
    r"D:\data\image_data\weather_image_recognition\rain\1084.jpg",
    r"D:\data\image_data\weather_image_recognition\rain\1824.jpg",
    r"D:\data\image_data\weather_image_recognition\rain\1090.jpg",
    r"D:\data\image_data\weather_image_recognition\rain\1013.jpg"
]


rows, cols = 1, len(image_paths)
fig, axes = plt.subplots(rows, cols, figsize=(20, 5))

for ax, path in zip(axes.ravel(), image_paths):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    ax.set_title(path.split("\\")[-1], fontsize=10)  # Dateiname als Titel
    ax.axis("off")

plt.tight_layout()
plt.show()
