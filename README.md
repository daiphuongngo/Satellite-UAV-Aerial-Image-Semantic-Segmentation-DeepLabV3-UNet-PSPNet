# Satellite-UAV-Aerial-Image-Semantic-Segmentation-DeepLabV3-UNet-ENet-PSPNet


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI S-89 **Deep Learning** (Python)

## Professors: Dmitry V. Kurochkin, PhD

Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

## Timeline: June 23 - August 12, 2025

## 🌍 Satellite & UAV Aerial Image Semantic Segmentation

> A deep learning project for multi-class semantic segmentation on aerial imagery using **PSPNet**, **UNet**, and **DeepLabV3+**, applied to three datasets: UAVID, modified Bhuvan Land Cover, and semantic tile datasets.

---

### 📂 Directory Structure

```bash
.
project_root_on_Drive/
├── modified_uavid_dataset/
│   ├── train_data/
│   │   ├── Labels/                ← Original MUD masks
│   │   └── Labels_new/            ← Recolorized MUD masks
│   └── val_data/
│       ├── Labels/                ← Original MUD masks (validation)
│       └── Labels_new/            ← Recolorized MUD masks (validation)
│
├── Semantic segmentation dataset/
│   └── Tile 1/
│       ├── masks/                 ← Original SIM masks
│       └── masks_new/             ← Recolorized SIM masks
│
└── Land Cover Classification Bhuvan Satellite Data/
    ├── train_mask/                ← Original SSAI-style masks (train)
    ├── train_mask_new/            ← Recolorized SSAI masks (train)
    ├── test_mask/                 ← Original SSAI-style masks (test)
    └── test_mask_new/             ← Recolorized SSAI masks (test)
```

---

### 🚀 Model Architectures

* ✅ **UNet**: Custom lightweight encoder-decoder architecture for pixel-wise segmentation.
* ✅ **DeepLabV3+**: Uses atrous spatial pyramid pooling (ASPP) and encoder-decoder modules for better boundary capture.
* ✅ **PSPNet**: Pyramid Scene Parsing Network that handles global context using pyramid pooling module.
* ✅ **SAM2 Support in WherobotsAI Raster Inference**: Integrate support for Meta AI's Segment Anything Model 2 (SAM2) and Google DeepMind's OWLv2 models for text-prompted inference, and create 2 new Raster Inference functions (RS_Text_to_BBoxes and RS_Text_to_Segments) for converting text prompts to segmentation and object detection results.
---

### 🧠 Training Pipeline

```python
# train.py
from models import unet, deeplabv3plus, pspnet
from utils.data_loader import load_dataset
from utils.metrics import iou_score, plot_cm
from utils.plot_utils import plot_sample_predictions

# Choose dataset and model
dataset = load_dataset('uavid')
model = unet.build(input_shape=(256, 256, 3), num_classes=6)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', iou_score])
model.fit(train_ds, epochs=50, validation_data=val_ds)
model.save("models/unet_uavid.h5")
```

---

### 📊 Metrics & Visualization

* Mean IoU
* Pixel accuracy
* Confusion matrix
* Sample prediction overlay

---

### 🗂️ Datasets

| Dataset           | Classes                    | Source                   |
| ----------------- | -------------------------- | ------------------------ |
| UAVID             | Building, Road, Tree, etc  | UAV urban scenes         |
| Bhuvan Land Cover | Land, Water, Urban, Forest | Indian satellite imagery |
| Semantic Tiles    | Custom semantic tiles      | Dubai aerial data      |

---

### 📦 Requirements

```txt
tensorflow>=2.9
opencv-python
scikit-image
matplotlib
numpy
```

---

### 📌 Future Work

* Integrate **transformer-based** segmentation (SegFormer)
* Apply **SAM** (Segment Anything Model) as pre-segmentation
* Deploy to web app using **Streamlit** or **Gradio**

---

![SSAI photo 1](https://github.com/user-attachments/assets/b0b336cb-c934-4d97-882c-ff6ddb936437)

![MUD photo 1](https://github.com/user-attachments/assets/afa294e7-24d4-41b4-8c86-f6f64aaf0039)

![SIM photo 1](https://github.com/user-attachments/assets/1ac2de13-295e-4909-a70d-feee7a9b29d8)


## Project Goal and Problem Statement


## Expected Results


## Application Overview and Technologies used 

## Processing Pipeline

