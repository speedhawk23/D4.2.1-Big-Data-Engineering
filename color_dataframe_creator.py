import pandas as pd
import math

# 1. Farbdaten (hier erstmal Dummy-Daten)
color_data = [
    {"img": "image-1.jpg", "color_1": (120, 80, 60), "color_2": (130, 90, 70), "color_3": (140, 100, 80)},
    {"img": "image-2.jpg", "color_1": (200, 50, 60), "color_2": (210, 60, 70), "color_3": (220, 70, 80)},
    {"img": "image-3.jpg", "color_1": (50, 100, 150), "color_2": (60, 110, 160), "color_3": (70, 120, 170)},
]

# 2. DataFrame erstellen
df = pd.DataFrame(color_data)
print(df.head())

# 3. DataFrame speichern
df.to_csv('farben.csv', index=False)
print("✅ CSV-Datei 'farben.csv' wurde erstellt!")

# 4. Vergleichsfunktion
def color_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

# 5. Beispiel
color1 = df.loc[0, 'color_1']
color2 = df.loc[1, 'color_1']

# Umwandeln, wenn Farben als Text gespeichert sind
if isinstance(color1, str):
    color1 = eval(color1)
if isinstance(color2, str):
    color2 = eval(color2)

distance = color_distance(color1, color2)
print(f"🎯 Die Farbdistanz zwischen Bild 1 und Bild 2 ist: {distance}")
