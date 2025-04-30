from colorthief import ColorThief
import os

cwd = os.getcwd()
filepath = os.path.join(cwd, "testbilder")

# Ändere color_data zu einer Liste
color_data = []

for i in range(705):
    img_path = os.path.join(filepath, f"image-{i + 1}.jpg")
    if os.path.exists(img_path):  # Überprüfe, ob das Bild existiert
        ct = ColorThief(img_path)
        dominant_color = ct.get_color(quality=1)
        palette = ct.get_palette(color_count=3)  # Hole die ersten 3 Farben aus der Palette

        # Füge die Daten in die Liste ein
        color_data.append({
            "img": f"image-{i + 1}.jpg",
            "color_1": palette[0] if len(palette) > 0 else None,
            "color_2": palette[1] if len(palette) > 1 else None,
            "color_3": palette[2] if len(palette) > 2 else None
        })
    else:
        print(f"Bild {img_path} wurde nicht gefunden.")

# Optional: Zeige die ersten Einträge an
for entry in color_data[:5]:  # Zeige die ersten 5 Einträge
    print(entry)