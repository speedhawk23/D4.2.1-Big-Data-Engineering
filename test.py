from colorthief import *
from matplotlib import pyplot as plt
import os
import colorsys
import pandas as pd

cwd = os.getcwd()
filepath = os.path.join(cwd, "testbilder")

color_data = {}

for i in range(705):
    globals()[f"img_{i + 1}"] = os.path.join(filepath, f"image-{i + 1}.jpg")
    ct = ColorThief(globals()[f"img_{i + 1}"])
    dominant_color = ct.get_color(quality=1)
    palette = ct.get_palette(color_count=2)
    color_data.append({
    "img": globals()[f"img_{i + 1}"], "color_1": palette[0] if len(palette) > 0 else None, "color_2": palette[1] if len(palette) > 1 else None, "color_3": palette[2] if len(palette) > 2 else None})


"""
#print(img_5)
img = Image.open(img_8)
plt.imshow(img)
plt.show()


ct = ColorThief(img_8)
dominant_color = ct.get_color(quality=1)

plt.imshow([[dominant_color]])
plt.show()

palette = ct.get_palette(color_count=2)
plt.imshow([[palette[i]for i in range (3)]])
plt.show()

for color in palette:
    #print(color)
    print(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")

color_dict = {"Image Name" = img}


    img color_1 cololr_2 color3
0
1
2
3
"""