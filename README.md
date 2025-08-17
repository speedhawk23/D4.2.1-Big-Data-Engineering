# Big Data Image Recommender

Dieses Projekt implementiert ein System zur Bildähnlichkeitssuche für große Datensätze.  
Es kombiniert klassische Methoden (HSV-Histogramme, HOG) mit Deep-Learning-Embeddings (EfficientNet-B0) und ermöglicht eine schnelle und skalierbare Suche nach ähnlichen Bildern. Zusätzlich können die Verfahren kombiniert werden, um robustere Ergebnisse zu erzielen.

## Funktionen
- Indexierung und Verwaltung von Bilddaten mit SQLite
- Farbähnlichkeit über HSV-Histogramme
- Strukturanalyse mit Histogram of Oriented Gradients (HOG)
- Deep-Learning-Embeddings mit PyTorch und FAISS
- Approximate Nearest Neighbor Suche mit HNSW
- Kombination der Verfahren für verbesserte Ergebnisse
- Analyse und Visualisierung großer Bildmengen mit PCA und UMAP

## Projektstruktur
- `bigdata_analysis/` – Skripte für Visualisierung und Dimensionality Reduction  
- `features/embed/` – Feature-Extraktion und Embedding-Pipeline  
- `legacy/` – ältere Versionen und Prototypen  
- `tests/` – automatisierte Tests  
- `image_indexer.py` – Indexierung und Datenbankverwaltung  
- `image_recommender_hsv.py` – Suche mit HSV-Features  
- `image_recommender_hog.py` – Suche mit HOG-Features  
- `image_recommender_embeddings.py` – Suche mit Embeddings  
- `image_recommender_combo.py` – kombinierte Suche  

## Installation
```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
```

## Nutzung
Indexierung starten:
```bash
python image_indexer.py --stage index --path /path/to/images
```

HSV-Suche:
```bash
python image_recommender_hsv.py --stage hsv --stage search --query query.jpg
```

Embedding-Suche mit HNSW:
```bash
python image_recommender_embeddings.py --stage embed --stage build_index_hnsw
python image_recommender_embeddings.py --stage search_hnsw --query query.jpg
```

## Anforderungen
- Python 3.9+
- OpenCV
- NumPy
- PyTorch
- FAISS
- SQLite

## Dokumentation
Weitere Details sind in der [Projektdokumentation](Dokumentation_Big_data.pdf) beschrieben.
