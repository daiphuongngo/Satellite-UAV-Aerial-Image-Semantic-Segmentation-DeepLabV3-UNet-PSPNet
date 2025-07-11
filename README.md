# Satellite-UAV-Aerial-Image-Semantic-Segmentation-DeepLabV3-UNet-ENet-PSPNet


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI S-89 **Deep Learning** (Python)

## Professor: Dmitry V. Kurochkin, PhD

Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

## Timeline: June 23 - August 6, 2025

## 🌍 Satellite & UAV Aerial Image Semantic Segmentation

A deep learning project for multi-class semantic segmentation on aerial imagery using **PSPNet**, **UNet** and **DeepLabV3+**, applied to three datasets: UAVID, modified Bhuvan Land Cover and semantic tile datasets.

---
### Executive Summary

This project tackles multi-class semantic segmentation of high-resolution aerial and satellite imagery, enabling applications like land cover classification, smart city planning and disaster monitoring. I built deep learning pipelines using UNet, DeepLabV3+ and PSPNet on diverse datasets (UAVID, Bhuvan Land Cover and Dubai SIM).

Key results:
- Achieved pixel-level accuracy > 96% approximately
- Effective in distinguishing fine-grained urban features like roundabouts, buildings, and road types
- Built multi-format dataset loaders, recolorization utilities and full training pipelines

---

### Project Overview

Semantic segmentation is a critical task in remote sensing and environmental monitoring. My project focuses on classifying satellite pixels into 8 predefined land cover classes based off the satellite imagery of Dubai using a supervised learning approach. My scope is  to implement and train these  three semantic segmentation models:
•	U-Net: this is a new approach as I researched with a compact encoder-decoder network widely used in medical and satellite image segmentation.

•	DeepLabV3+: this is my top two most preferred advanced model using atrous spatial pyramid pooling (ASPP) and encoder-decoder refinement.

•	PSPNet: this is my third choice of triangle-like comparison, which is a powerful model leveraging pyramid pooling for global context aggregation.

Towards the end of my project, I will benchmark these models on accuracy, class-wise metrics, and performance trade-offs.

--- 
### Problem Statement

I found that semantic segmentation of satellite and UAV (unmanned aerial vehicle) images is crucial for understanding and classifying land use and land cover (LULC), identifying environmental patterns, detecting urban growth (e.g. traffic control, building construction, water source maintenance) and managing disaster response (e.g., flood mapping, wildfire detection). My project aims to build a robust deep learning pipeline using multiple architectures, such asU-Net, DeepLabV3+, and PSPNet, for segmenting complex and diverse classes in satellite and UAV imagery.

When comparing to standard computer vision tasks, I can observe that aerial imagery prompts several unique challenges that I will classify as follows:

•	High spatial variability: Based off the ground objects on my selected data sources , such as buildings, roads, water bodies, and vegetation vary widely in scale, shape, and texture depending on resolution, region, and angles of captured images.

•	Inter-class similarity: A significant challenge in handling the data is about certain land types (e.g., bare soil vs. urban concrete, forestry or tree lines vs. grassland) can appear visually similar, increasing the risk of misclassification.

•	Image artifacts and noise: UAV-captured images can suffer from motion blur, variable lighting conditions, or occlusion due to atmospheric interference. Their angles of images taken can also increase the diversity of objects and artifacts detected. Meanwhile, although satellite-captured images offer a consistent angle from above atmosphere, they can experience obstacles like clouds and shadow that diminish the ground mass coverage and full sizes of objects (such as buildings and cars).

•	Large image sizes: Satellite imagery often comes in high resolution, demanding significant computational resources for training and inference. This can easily exceed the standard and complimentary 15GB storage of one Google Drive so I had to purchase a better Drive plan for storing without affecting other purposes.

•	Class imbalance: In many segmentation datasets that I chose, majority classes (e.g., background, vegetation (including grassland, forestry/tree lines and angriculture) heavily outweigh minority classes (e.g., water edges, cars, pools, human, fire zones), resulting in biased model performance.

Therefore, to address these challenges, I will use the following approaches:

•	Data augmentation: This techniques offers rotation, scaling, horizontal flips, color jittering, and noise injection are applied to enhance model generalization across environmental conditions.

•	Indexed masks with original colors: The dataset uses RGB-encoded masks where each color corresponds to a distinct class. No manual color maps are used. My models are trained to learn directly from RGB-based class masks.

•	Transfer learning: I also employ retrained encoders (e.g., ResNet backbones) to reduce training time and improve convergence with limited labeled data.

•	Multi-metric evaluation: Instead of relying solely on pixel-level accuracy, my evaluation includes precision, recall, F1-score, and class-wise Intersection over Union (IoU) for each model assessment and deployment.

My goal of this project is to compare the effectiveness of different deep learning architectures in segmenting semantic classes across different geographic and environmental contexts, producing accurate segmentation masks for visual analytics, geospatial modeling, and decision support.

---

### Data Exploration, Modelling, Algorithm and Evaluation Strategy
My semantic segmentation pipeline was built on RGB satellite and UAV images with pixel-level RGB masks representing various land-use classes. My project follows a supervised learning framework where the model learns to classify each pixel into one of reduced 6 classes based on the color of the ground truth mask. My pipeline includes the following components:

#### Data Pipeline & Augmentation

•	Augmented images and their corresponding RGB masks were resized to 256×256 to balance detail preservation and memory efficiency.

•	Augmentation included random rotations, flips, zooms, contrast enhancements, and noise addition.

•	Ground truth masks maintain original RGB values without converting to class indices, ensuring human-interpretable segmentation outputs.

#### Data Understanding
My project utilizes semantic segmentation datasets derived from satellite and UAV imagery, each containing RGB images and corresponding color-encoded segmentation masks. The original datasets include:

•	SSAI (Semantic Segmentation of Aerial Imagery on Dubai - baseline): A structured dataset with well-defined semantic classes and unique RGB values per class.

•	SIM (Land Cover Classification: Bhuvan Satellite Data): A rich set of labeled satellite imagery with differing label colors and class definitions.

•	MUD (Modified UAVid Dataset): High-resolution UAV-captured scenes with detailed object annotations.

While the three datasets cover diverse geographic regions and object types, their class definitions and color encodings were initially inconsistent. For instance, SIM used bright visual colors like pure yellow for agriculture, and MUD had separate labels for “moving car,” “static car,” and “human,” whereas SSAI offered unified labels for high-level object categories.
To address this inconsistency, I adopted SSAI’s class-color structure as the common reference palette for unified training simplification, better model generalization and consistent assessment. I implemented a color normalization function using KDTree-based nearest-neighbor color matching, allowing masks from SIM and MUD to be recolored into the SSAI palette. This enabled a standardized pipeline for:

•	Preprocessing

•	Model input preparation

•	Visual inspection

•	Evaluation and class-based performance comparison

#### Unified Class Mapping

My recoloring process grouped visually and semantically similar classes across datasets


|Original Label	|Unified SSAI Class	|SSAI RGB Color|
|-|-|-|
|Agriculture, Forest, Tree, Low vegetation	|Vegetation	|(254, 221, 58)|
|Building	|Building	|(60, 16, 152)|
|Road	|Road	|(110, 193, 228)|
|Water	|Water	|(226, 169, 41)|
|Human, Moving Car, Static Car	|Object	|(232, 98, 60)|
|Unlabeled, Background clutter	|Unlabeled	|(0, 0, 0)|

#### 📂 Directory Structure
My working directory was organized for seamless integration with the TensorFlow Dataset (TFDS) API and batched training:

•	/augmented_images/: Contains augmented and resized RGB input images (256×256 resolution).

•	/augmented_masks_indexed/: Contains segmentation masks where each pixel’s RGB color directly maps to a semantic class (based on the unified SSAI palette).

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

### Model Architectures

* **UNet**: Custom lightweight encoder-decoder architecture for pixel-wise segmentation.
* **DeepLabV3+** (assumed to be best model): Uses atrous spatial pyramid pooling (ASPP) and encoder-decoder modules for better boundary capture.
* **PSPNet**: Pyramid Scene Parsing Network that handles global context using pyramid pooling module.

---

### Training Pipeline

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
| Semantic Tiles (main)    | Custom semantic tiles      | Dubai aerial data      |
| UAVID  (3rd supporting)             | Building, Road, Tree, etc  | UAV urban scenes         |
| Bhuvan Land Cover (2nd supporting) | Land, Water, Urban, Forest | Indian satellite imagery |

---

### Requirements

```txt
tensorflow>=2.9
opencv-python
scikit-image
matplotlib
numpy
```

---

### Future Work

* Integrate **transformer-based** segmentation (SegFormer)
* Apply built model's prediction on new satellite images
* etc

---

![SSAI photo 1](https://github.com/user-attachments/assets/b0b336cb-c934-4d97-882c-ff6ddb936437)

![MUD photo 1](https://github.com/user-attachments/assets/afa294e7-24d4-41b4-8c86-f6f64aaf0039)

![SIM photo 1](https://github.com/user-attachments/assets/1ac2de13-295e-4909-a70d-feee7a9b29d8)

---
## Processing Pipeline


Here's my **comparison summary of U-Net, DeepLabV3+ and PSPNet** based on my reported classification results, confusion matrices, and visual predictions:

---

### DeepLabV3+

![DeepLabV3+](https://github.com/user-attachments/assets/e6388423-8d75-437e-ac1f-45ef908ae5c6)

![DeepLabV3+ - predicted output 2](https://github.com/user-attachments/assets/36815bf6-dd92-4f0d-87d7-f77bf4c531ab)

![DeepLabV3+ - predicted output 1](https://github.com/user-attachments/assets/c29f0dd6-a2cb-4ed9-b640-377757b7e3a2)

---

### UNet

![UNet](https://github.com/user-attachments/assets/2a530c09-f852-48c4-84e3-6ac3067ba9be)

![UNet - predicted output 2](https://github.com/user-attachments/assets/e1f3fcbd-29e9-478b-8b88-70bbb88928b3)

![UNet - predicted output 1](https://github.com/user-attachments/assets/edd2e254-263a-4959-a27d-940b6cfe11fa)

---

### PSPNet

![PSP Net](https://github.com/user-attachments/assets/2a0b475f-6055-4d84-8d24-5a7d576aaba7)

![PSP Net - predicted output 2](https://github.com/user-attachments/assets/778af281-a7c6-4fb3-97d2-b5f590358867)

![PSP Net - predicted output 1](https://github.com/user-attachments/assets/4a5e2f25-f859-4960-a985-a7f7a85bc3d3)

---

### 🔢 **Quantitative Performance Summary**

| Model          | Accuracy   | Precision  | Recall     | F1 Score   | Notes                                           |
| -------------- | ---------- | ---------- | ---------- | ---------- | ----------------------------------------------- |
| **U-Net**      | 0.9446     | 0.9446     | 0.9446     | 0.9446     | Best balance across classes                     |
| **DeepLabV3+** | 0.9507     | 0.9511     | 0.9507     | 0.9508     | Highest accuracy and IoU                        |
| **PSPNet**     | 0.8572     | 0.8540     | 0.8572     | 0.8528     | Struggles on complex boundaries (e.g., class 3) |

---

### **Class-wise Comparison (Validation Set)**

| Class | Description | U-Net F1 | DeepLabV3+ F1 | PSPNet F1 |
| ----- | ----------- | -------- | ------------- | --------- |
| 0     | Black       | 0.98     | 0.98          | 0.93      |
| 1     | Purple      | 0.93     | 0.95          | 0.76      |
| 2     | Blue        | 0.95     | 0.96          | 0.89      |
| 3     | Violet      | 0.86     | 0.84          | 0.55      |
| 4     | Yellow      | 0.92     | 0.94          | 0.84      |
| 5     | Orange      | 0.99     | 0.99          | 0.96      |
| 7     | Gray        | 0.90     | 0.93          | 0.80      |

> 💡 **DeepLabV3+** consistently outperforms in most classes, especially in larger and more continuous regions (e.g., class 2, 5).
> **PSPNet** struggles especially on fragmented classes like 3 (urban/road class?) and 7 (small structures?).

---

### **Visual Segmentation Results**

| Model          | Observations                                                                                               |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| **U-Net**      | Very smooth and accurate masks. Strong in edges and small classes.                                         |
| **DeepLabV3+** | Best global structure preservation. Sharp boundaries and excellent class separation.                       |
| **PSPNet**     | Blurry or blocky predictions. Misses fine details. Good for coarse segmentation, less for fine structures. |

> 📸 Example comparison shown in your predictions: DeepLabV3+ captured roads and roundabouts best, followed by U-Net. PSPNet was visibly noisier and more blocky in those regions.

---

### **Scenario-based Model Application**

| Scenario                               | Recommended Model                                                         |
| -------------------------------------- | ------------------------------------------------------------------------- |
| **High accuracy & boundary precision** | ✅ DeepLabV3+                                                             |
| **Lightweight and fast**               | ✅ U-Net                                                                  |
| **Coarse terrain or large objects**    | ✅ PSPNet (only if compute is limited or architecture simplicity matters) |

---



## Model Comparison Summary
```markdown
| Model        | Accuracy | Precision | Recall | F1 Score |
|--------------|----------|-----------|--------|----------|
| U-Net        | 94.46%   | 94.46%    | 94.46% | 94.46%   |
| DeepLabV3+   | 95.07%   | 95.11%    | 95.07% | 95.08%   |
| PSPNet       | 85.72%   | 85.40%    | 85.72% | 85.28%   |
```
## Class-wise F1 Score (Validation Set)
```markdown
| Class | Description | U-Net  | DeepLabV3+ | PSPNet |
|-------|-------------|--------|------------|--------|
| 0     | Background  | 0.98   | 0.98       | 0.93   |
| 1     | Class 1     | 0.93   | 0.95       | 0.76   |
| 2     | Class 2     | 0.95   | 0.96       | 0.89   |
| 3     | Class 3     | 0.86   | 0.84       | 0.55   |
| 4     | Class 4     | 0.92   | 0.94       | 0.84   |
| 5     | Class 5     | 0.99   | 0.99       | 0.96   |
| 7     | Class 7     | 0.90   | 0.93       | 0.80   |
```

## Model Training Speed and Complexity
```markdown


| Model        | Total Epochs | Avg Time per Epoch | Total Training Time  | Total Params     | Size (MB) | Notes                                      |
|--------------|--------------|--------------------|----------------------|------------------|-----------|--------------------------------------------|
| U-Net        | 100          | ~57 sec            | ~1.6 hrs             | 1.94M            | ~7.41 MB  | Fastest; lightest; best for small datasets |
| DeepLabV3+   | 100          | ~2.3 min           | ~3.8 hrs             | 11.85M           | ~45.22 MB | Best accuracy; ResNet50 + ASPP             |
| PSPNet       | 100          | ~1.8 min           | ~3.0 hrs             | 24.86M           | ~94.82 MB | Heaviest; large memory footprint           |

```

- **U-Net** has the smallest number of parameters, ideal for faster iteration or limited hardware.
- **DeepLabV3+** strikes a strong balance between accuracy and efficiency.
- **PSPNet** is more complex, requiring more VRAM but still achieves competitive results.



