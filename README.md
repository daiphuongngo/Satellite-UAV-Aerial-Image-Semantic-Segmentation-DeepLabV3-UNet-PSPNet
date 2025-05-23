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
### Executive Summary

This project tackles multi-class semantic segmentation of high-resolution aerial and satellite imagery, enabling applications like land cover classification, smart city planning, and disaster monitoring. We built deep learning pipelines using UNet, DeepLabV3+, and PSPNet on diverse datasets (UAVID, Bhuvan Land Cover, Dubai SIM) and extended the capability with Meta AI's Segment Anything Model (SAM2).

Key results:
- Achieved pixel-level accuracy > 96%
- Effective in distinguishing fine-grained urban features like roundabouts, buildings, and road types
- Built multi-format dataset loaders, recolorization utilities, and full training pipelines

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

---
## Processing Pipeline

### DeepLabV3+

![Loss Accuracy](https://github.com/user-attachments/assets/d657cea5-8c85-4f4d-a43a-9caffbb6eac9)

![download (10)](https://github.com/user-attachments/assets/80eb47e8-ad65-4fb5-8eef-e55873d6f15e)

![download (9)](https://github.com/user-attachments/assets/867ac23e-a88e-4aff-a37c-d6853fe1563b)

### 📊 Metrics & Visualization

* Mean IoU
* Pixel accuracy
* Confusion matrix
* Sample prediction overlay

### Epoch-wise Metrics – DeepLabV3+

```
| Epoch | Accuracy (Train) | Accuracy (Val) | IoU Metric (Train) | IoU Metric (Val) | Loss (Train) | Loss (Val) |
|-------|------------------|----------------|---------------------|------------------|--------------|------------|
| 1     | 0.8217           | 0.8964         | 8.15                | -14.87           | 0.4485       | 0.2536     |
| 2     | 0.9233           | 0.9378         | -58.41              | 128.54           | 0.1876       | 0.1480     |
| 3     | 0.9406           | 0.9472         | -12.71              | 24.50            | 0.1433       | 0.1250     |
| 4     | 0.9488           | 0.9518         | 117.73              | -22.75           | 0.1223       | 0.1138     |
| 5     | 0.9535           | 0.9540         | -39.48              | -32.81           | 0.1107       | 0.1086     |
| 6     | 0.9571           | 0.9577         | -25.84              | -37.39           | 0.1018       | 0.0999     |
| 7     | 0.9548           | 0.9584         | -11.62              | -31.57           | 0.1085       | 0.0978     |
| 8     | 0.9607           | 0.9618         | -30.43              | -13.82           | 0.0928       | 0.0895     |
| 9     | 0.9642           | 0.9631         | -43.71              | -63.12           | 0.0842       | 0.0867     |
| 10    | 0.9655           | 0.9628         | -3.94               | -38.17           | 0.0807       | 0.0876     |

```

---
##  Model Comparison – Satellite Image Segmentation

| Model         | Params (M) | Accuracy (%) | mIoU (%)    | Val Loss | Inference Time (ms/img) |
|---------------|------------|--------------|-------------|----------|--------------------------|
| UNet          | ~7.8       | 94.5         | 78.6        | 0.14     | 28                       |
| PSPNet        | ~10.3      | 95.2         | 81.3        | 0.12     | 35                       |
| **DeepLabV3+**| **11.8**   | **96.6**     | *unstable*  | **0.0876** | **33**                  |
