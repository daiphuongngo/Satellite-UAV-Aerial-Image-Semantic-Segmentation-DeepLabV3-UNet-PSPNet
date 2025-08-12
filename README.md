# Satellite-UAV-Aerial-Image-Semantic-Segmentation-DeepLabV3-UNet-PSPNet


![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master, Data Science**

## CSCI S-89 **Deep Learning** (Python)

## Professor: Dmitry V. Kurochkin, PhD

Senior Research Analyst, Faculty of Arts and Sciences Office for Faculty Affairs, Harvard University

## Author: **Dai-Phuong Ngo (Liam)**

## Timeline: June 23 - August 6, 2025

## 🌍🛰🛸 Satellite & UAV Aerial Image Semantic Segmentation

My deep learning project for multi-class semantic segmentation on aerial imagery using **PSPNet**, **UNet** and **DeepLabV3+**, applied to three datasets: UAVID, modified Bhuvan Land Cover and semantic tile datasets.

Youtube: https://www.youtube.com/watch?v=OpkUmdsbYrE

---
### ABSTRACT 

My project for **CSCI S-89 Deep Learning** is motivated by my prior research on brain MRI segmentation in CSCI E-25, which sparked my interest in applying semantic segmentation to geospatial challenges. In this project, I focus on **high-resolution satellite and UAV imagery** to classify each pixel into one of eight land cover classes, using **augmented RGB images as inputs** and **class-indexed segmentation masks as outputs**. The dataset used originates from publicly available **land cover segmentation datasets** commonly used in remote sensing and aerial mapping tasks. It includes **several thousand image–mask pairs**, where each observation is a 256×256 RGB tile paired with its corresponding 8-class label mask.

To evaluate the problem, I trained and compared **three deep learning architectures**: U-Net, DeepLabV3+, and PSPNet. The models were benchmarked on classification accuracy, Intersection over Union (IoU), loss metrics, model size, and qualitative segmentation performance. Training was performed using **Google Colab Pro+**, utilizing an **NVIDIA A100 GPU**, with an estimated **50 compute units per model** and **100 epochs per run**.

Preliminary results suggest that **DeepLabV3+** offers the best trade-off between precision and generalization, while **U-Net** is the most efficient in constrained environments. **PSPNet**, despite its large parameter size, showed weaker performance. This project underscores the importance of balancing model complexity, compute resources, and segmentation quality in real-world remote sensing applications.

---
### Executive Summary

This project tackles multi-class semantic segmentation of high-resolution aerial and satellite imagery, enabling applications like land cover classification, smart city planning and disaster monitoring. I built deep learning pipelines using UNet, DeepLabV3+ and PSPNet on diverse datasets (UAVID, Bhuvan Land Cover and Dubai SIM).

Key results:
- Achieved pixel-level accuracy > 96% approximately
- Effective in distinguishing fine-grained urban features like roundabouts, buildings. water streams and road types
- Built multi-format dataset loaders, recolorization utilities and full training pipelines

---

### Project Overview

Semantic segmentation is a critical task in remote sensing and environmental monitoring. My project focuses on classifying satellite pixels into 8 predefined land cover classes based off the satellite imagery of Dubai using a supervised learning approach. My scope is  to implement and train these  three semantic segmentation models:
-	**U-Net**: this is a new approach as I researched with a compact encoder-decoder network widely used in medical and satellite image segmentation.

-	**DeepLabV3+**: this is my top two most preferred advanced model using atrous spatial pyramid pooling (ASPP) and encoder-decoder refinement.

-	**PSPNet**: this is my third choice of triangle-like comparison, which is a powerful model leveraging pyramid pooling for global context aggregation.

Towards the end of my project, I will benchmark these models on accuracy, class-wise metrics and performance trade-offs.

--- 
### Problem Statement

I found that semantic segmentation of satellite and UAV (unmanned aerial vehicle) images is crucial for understanding and classifying land use and land cover (LULC), identifying environmental patterns, detecting urban growth (e.g. traffic control, building construction, water source maintenance) and managing disaster response (e.g., flood mapping, wildfire detection, object detection). My project aims to build a robust deep learning pipeline using multiple architectures, such asU-Net, DeepLabV3+, and PSPNet, for segmenting complex and diverse classes in satellite and UAV imagery.

When comparing to standard computer vision tasks, I can observe that aerial imagery prompts several unique challenges that I will classify as follows:

-	**High spatial variability**: Based off the ground objects on my selected data sources , such as buildings, roads, water bodies and vegetation vary widely in scale, shape, and texture depending on resolution, region, and angles of captured images.

-	**Inter-class similarity**: A significant challenge in handling the data is about certain land types (e.g., bare soil vs. urban concrete, forestry or tree lines vs. grassland) can appear visually similar, increasing the risk of misclassification.

-	**Image artifacts and noise**: UAV-captured images can suffer from motion blur, variable lighting conditions, or occlusion due to atmospheric interference. Their angles of images taken can also increase the diversity of objects and artifacts detected. Meanwhile, although satellite-captured images offer a consistent angle from above atmosphere, they can experience obstacles like clouds and shadow that diminish the ground mass coverage and full sizes of objects (such as buildings and cars).

-	**Large image sizes**: Satellite imagery often comes in high resolution, demanding significant computational resources for training and inference. This can easily exceed the standard and complimentary 15GB storage of one Google Drive so I had to purchase a better Drive plan for storing without affecting other purposes.

-	**Class imbalance**: In many segmentation datasets that I chose, majority classes (e.g., background, vegetation (including grassland, forestry/tree lines and angriculture) heavily outweigh minority classes (e.g., water edges, cars, pools, human, fire zones), resulting in biased model performance.

Therefore, to address these challenges, I will use the following approaches:

-	**Data augmentation**: This techniques offers rotation, scaling, horizontal flips, color jittering and noise injection are applied to enhance model generalization across environmental conditions.

-	**Indexed masks with original colors**: The dataset uses RGB-encoded masks where each color corresponds to a distinct class. No manual color maps are used. My models are trained to learn directly from RGB-based class masks.

-	**Transfer learning**: I also employ retrained encoders (e.g., ResNet backbones) to reduce training time and improve convergence with limited labeled data.

-	**Multi-metric evaluation**: Instead of relying solely on pixel-level accuracy, my evaluation includes precision, recall, F1-score, and class-wise Intersection over Union (IoU) for each model assessment and deployment.

My goal of this project is to compare the effectiveness of different deep learning architectures in segmenting semantic classes across different geographic and environmental contexts, producing accurate segmentation masks for visual analytics, geospatial modeling and decision support.

---

### Data Exploration, Modelling, Algorithm and Evaluation Strategy
My semantic segmentation pipeline was built on RGB satellite and UAV images with pixel-level RGB masks representing various land-use classes. My project follows a supervised learning framework where the model learns to classify each pixel into one of reduced 6 classes based on the color of the ground truth mask. My pipeline includes the following components:

#### Data Pipeline & Augmentation

-	Augmented images and their corresponding RGB masks were resized to 256×256 to balance detail preservation and memory efficiency.

-	Augmentation included random rotations, flips, zooms, contrast enhancements and noise addition.

-	Ground truth masks maintain original RGB values without converting to class indices, ensuring human-interpretable segmentation outputs.

#### Data Understanding

My project utilizes semantic segmentation datasets derived from satellite and UAV imagery, each containing RGB images and corresponding color-encoded segmentation masks. The original datasets include:

•	**SSAI** (Semantic Segmentation of Aerial Imagery on Dubai - baseline): A structured dataset with well-defined semantic classes and unique RGB values per class.

•	**SIM** (Land Cover Classification: Bhuvan Satellite Data): A rich set of labeled satellite imagery with differing label colors and class definitions.

•	**MUD** (Modified UAVid Dataset): High-resolution UAV-captured scenes with detailed object annotations.

While the three datasets cover diverse geographic regions and object types, their class definitions and color encodings were initially inconsistent. For instance, SIM used bright visual colors like pure yellow for agriculture, and MUD had separate labels for “moving car,” “static car,” and “human,” whereas SSAI offered unified labels for high-level object categories.

#### 🗂️ Datasets

| Dataset           | Classes                    | Source                   |
| ----------------- | -------------------------- | ------------------------ |
| Semantic Tiles (main)    | Custom semantic tiles      | Dubai aerial data      |
| UAVID  (3rd supporting)             | Building, Road, Tree, etc  | UAV urban scenes         |
| Bhuvan Land Cover (2nd supporting) | Land, Water, Urban, Forest | Indian satellite imagery |


To address this inconsistency, I adopted SSAI’s class-color structure as the common reference palette for unified training simplification, better model generalization and consistent assessment. I implemented a color normalization function using KDTree-based nearest-neighbor color matching, allowing masks from SIM and MUD to be recolored into the SSAI palette. This enabled a standardized pipeline for:

-	Preprocessing

-	Model input preparation

-	Visual inspection

-	Evaluation and class-based performance comparison

#### Unified Class Mapping

My recoloring process grouped visually and semantically similar classes across datasets:


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

-	/augmented_images/: Contains augmented and resized RGB input images (256×256 resolution).

-	/augmented_masks_indexed/: Contains segmentation masks where each pixel’s RGB color directly maps to a semantic class (based off the unified SSAI palette).

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
### Data Preprocessing 

The preprocessing pipeline involved the following steps:

•	**Mask recoloring**: All SIM and MUD masks were recolored using SSAI-compatible colors via norm_colors_quantized() and KDTree.

•	**Mask resizing**: Masks were resized with nearest-neighbor interpolation to preserve discrete class boundaries.

•	**Image resizing and padding**: All RGB images were padded and resized to 256×256 using resize_with_pad for training uniformity.

---
#### Final Class Set for Training

The final set of classes used for segmentation (all normalized to SSAI definitions):

|Class Index	|Class Name	|RGB Encoding|
|-|-|-|
|0	|Unlabeled	|(0, 0, 0)|
|1	|Building	|(60, 16, 152)|
|2	|Land	|(132, 41, 246)|
|3	|Road	|(110, 193, 228)|
|4	|Vegetation	|(254, 221, 58)|
|5	|Water	|(226, 169, 41)|
|6	|Object	|(232, 98, 60)|

This unified structure enabled cross-dataset training, robust model generalization, and consistent performance evaluation across diverse environments, such as urban landscapes, forest regions, and disaster-prone areas.

---
### Data Augmentation

To improve the generalization of all three semantic segmentation models, U-Net, DeepLabV3+, and PSPNet, I can not rely only the original datasets, I applied a set of robust image augmentations to increase dataset diversity and simulate real-world variability in satellite and UAV imagery. These augmentations were consistently applied to both images and their corresponding masks, ensuring spatial alignment was preserved.

I came up with my Augmentation Strategy. My augmentation pipeline included the following randomized transformations:

•	**Horizontal Flipping**: This technique introduced symmetry variability.

•	**Rotation (±20°)**: A slight rotation simulated different object orientations. This is true as object location and angles of view can be varied due to their own position on the ground and flying paths of satellite and UAV.

•	**Contrast Enhancement**: I also applied contrast jittering with random intensity factors (0.6–1.4) which is considered slight differences for image diversification.

•	**Autocontrast and Histogram Equalization**: I also enhanced image contrast automatically or balanced histogram intensities.

•	**Brightness Adjustment**: This modification helped me to adust lighting variability with random brightness scaling (0.7–1.3). This is true in reality as satellite and UAV may capture images in different lighting conditions.

With these augmentations, I could apply probabilistically per image-mask pair, ensuring diverse and unique outputs from each original sample.

Here are my Pipeline Characteristics as follows:

•	**Target Count**: I aimed for 10,000 augmented pairs across 8 different tile folders.

•	**Augmentation Coverage**: This coverage ensured that each valid image–mask pair was augmented uniformly to meet the total target.

•	**Input Resolution**: I also resized images and masks to 512×512 pixels using BILINEAR for images and NEAREST for masks.

•	**Frameworks**: I completed augmentation by using Python’s Pillow (PIL) library.

---

### Model Architectures

* **UNet**: This is a symmetric encoder-decoder architecture with skip connections. U-Net is effective in capturing fine-grained information (especially for spatial information) and performs well on smaller datasets with balanced augmentation.
* **DeepLabV3+** (assumed to be best model): This is a state-of-the-art architecture using atrous (dilated) convolutions and encoder-decoder structure with a deep backbone (e.g., Xception or MobileNet) and atrous spatial pyramid pooling (ASPP). It helps me to excel in capturing multi-scale context and improving boundary accuracy.
* **PSPNet**: Pyramid Scene Parsing Network that handles global context using pyramid pooling module. This is another of my selective models which utilizes pyramid pooling modules to extract context at multiple scales. PSPNet is well-suited for complex scenes with varying object sizes, for example, captured images with multiple known and yet-unknown objects by satellite and UAV, and performs strongly on large-scale, high-resolution datasets.

---

### Training Configuration

To prioritize high-quality segmentation results while maintaining computational feasibility, I adopted a fixed but robust training configuration across all three models (U-Net, DeepLabV3+, and PSPNet). While exploring multiple optimizers, batch sizes or loss functions could potentially improve performance marginally, training each architecture separately on large augmented datasets for over 100 epochs is resource-intensive. Thus, I focused on settings that are widely recognized for producing stable and effective results in semantic segmentation tasks.

#### Training Parameters
•	**Optimizer**: Adam optimizer with a learning rate of 1e-4 was chosen for its adaptive learning properties and fast convergence. Adam consistently performs well in segmentation tasks without requiring extensive tuning.

•	**Loss Function**: SparseCategoricalCrossentropy(from_logits=True) was selected because the segmentation masks are class-indexed rather than one-hot encoded. This choice simplifies training and integrates well with TensorFlow's segmentation workflows.

•	Metrics:

o	**Accuracy**: While simple, it offers a high-level view of performance.

o	**Intersection over Union (IoU)**: A custom IoU metric was implemented to assess class-wise segmentation quality. This metric provides a more relevant and granular performance evaluation, particularly for imbalanced or spatially complex classes.

•	**Epochs**: Training was conducted for up to 100 epochs per model. In practice, early stopping could be introduced based on validation IoU to shorten training without sacrificing performance, especially for architectures that converge early.

•	**Batch Size**: A batch size of 8 was used. This size represents a trade-off between GPU memory efficiency and gradient stability across all three deep models on high-resolution data.

•	**Framework**: All training and evaluation were performed using TensorFlow 2.x and Keras, which offer high-level API simplicity alongside GPU acceleration and built-in monitoring.

---
### Evaluation Strategy

My evaluation strategy came from two general aspects:

•	**Quantitative**: Performance was monitored through validation metrics (IoU and accuracy), along with optional classification metrics such as confusion matrices.

•	**Qualitative**: Visual comparisons between predicted segmentation masks and ground truth were carried out to inspect object boundaries, class fidelity, and overall spatial coherence.

By selecting stable and proven training settings, I was able to focus on ensuring fair comparisons across the three architectures and on refining preprocessing, color normalization, and model outputs. While more extensive hyperparameter exploration is academically ideal, the practical constraints of large-scale augmented datasets made this approach the most reasonable within the project timeline.


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

> 📸 Example comparison shown in my predictions: DeepLabV3+ captured roads and roundabouts best, followed by U-Net. PSPNet was visibly noisier and more blocky in those regions.

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

## Ground Truth vs Prediction Quality Analysis for 6 Semantic Segmentation Models
---

### **Rank #1. UNet ver 5**

* **Observation:** Prediction maps closely replicate the ground truth. Small structures, roads, and boundaries are well preserved.
* **Strengths:** Sharp spatial boundaries, minimal noise, excellent balance between detail and structure.
* **Limitations:** Very few; slight artifacts in dense regions, but overall highly accurate.
* **Verdict:** Best performer visually, showing exceptional spatial fidelity for its compact size.
  
<img width="1174" height="407" alt="UNET ver 5 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/7f4b4dbc-bb8e-4f37-81e2-0bd992da7e77" />

---

### **Rank #2. UNet ver 6**

* **Observation:** Slightly less defined than UNet v5 but still delivers good spatial structure and class separation.
* **Strengths:** Regularization improves generalization slightly at the cost of sharpness.
* **Limitations:** Slightly blurrier class edges and lower fine detail recovery than UNet v5.
* **Verdict:** Strong second-place model, especially for regularized performance in real-world scenarios.

<img width="1174" height="407" alt="UNET ver 6 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/3b511991-2432-4d7c-a762-b9af32ef43db" />

---

### **3. DeepLabV3+ ver 4**

* **Observation:** Highly detailed and sharp segmentation. Boundaries and small classes are well preserved.
* **Strengths:** Captures detailed structures and edge continuity.
* **Limitations:** Signs of slight overfitting—minor inconsistencies in less prominent areas.
* **Verdict:** Strong visual performance, particularly good at segmenting complex shapes.

<img width="1174" height="407" alt="DeepLabV3+ ver 4 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/00a00fd6-205d-452e-864f-89a4b818048f" />

---

### **4. DeepLabV3+ ver 5**

* **Observation:** Very clean and generalized segmentation, though slightly smoother and less sharp than v4.
* **Strengths:** Excellent generalization; regularization avoids overfitting.
* **Limitations:** Slight drop in edge definition and minor structural blurring in tight areas.
* **Verdict:** A close follow-up to v4, better generalization but slightly less crisp visually.

<img width="1174" height="407" alt="DeepLabV3+ ver 5 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/f0930fb7-2bc2-40cb-b0bb-d14c338faee7" />

---

### **5. PSPNet ver 2**

* **Observation:** Moderate improvement over PSPNet v1. Still lacks fine structure detection and boundary clarity.
* **Strengths:** Some regularization effect helps smooth predictions.
* **Limitations:** Large parameter size doesn’t translate to visual clarity; still coarse and blocky.
* **Verdict:** Not ideal for fine-grained segmentation tasks despite being regularized.

<img width="1174" height="407" alt="PSPNET ver 2 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/84e37427-3a23-4c88-bc06-e7293814aa56" />

---

### **6. PSPNet ver 1**

* **Observation:** Coarse segmentation with high spatial noise. Many regions do not align with ground truth.
* **Strengths:** Can detect some large regions correctly.
* **Limitations:** Lacks precision and fine structure, poorly segmented edges.
* **Verdict:** Worst visual performance; suffers from architectural limitations.

<img width="1174" height="407" alt="PSPNET ver 1 - Ground Truth vs Prediction" src="https://github.com/user-attachments/assets/7eb16ef2-78dd-48db-80a3-46028463d21c" />


---

## Summary Ranking (Best to Worst)

| Rank	| Model	| Visual Accuracy	| Notes | 
|-|-|-|-|
| 1	| UNET ver 5	| 🟢 | Excellent	| Best balance of detail and clarity | 
| 2	| UNET ver 6	| 🟢 | Very Good	| Slightly softer than ver 5 | 
| 3	| DeepLabV3+ v4	| 🟢 | Good	| Strong semantic understanding | 
| 4	| DeepLabV3+ v5	| 🟡 | Moderate	| Slight degradation due to regularization | 
| 5	| PSPNET ver 2	| 🔴 | Weak	| Over-smoothing and coarse structure | 
| 6	| PSPNET ver 1	| 🔴 | Poor	| Blurry, fails to capture key patterns | 

---

### Lessons Learned

My project taught me the importance of model architecture choice, how regularization affects learning stability, and the trade-offs between performance and computational resources. Data augmentation proved critical for robust performance. Visual inspection is as important as metrics in segmentation tasks.

---

### Limitations and Future Work

Limitations include class imbalance and the use of only 2D imagery. Future directions include using multi-spectral data, class-weighted loss functions, and deploying models with TensorRT or ONNX for real-time inference. I also plan to experiment with hybrid architectures like U-Net++ or SegFormer."

--- 

## Conclusion

My project demonstrates how deep learning can be leveraged to perform pixel-accurate classification on satellite and UAV imagery. U-Net and DeepLabV3+ models are well-suited for this task, each excelling under different constraints. This experience deepened my understanding of real-world applications of deep learning.

---

## References

### 1. Core Methodologies and Architectures

1.	U-Net
   
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In MICCAI, LNCS vol. 9351, pp. 234–241. Springer.

https://arxiv.org/abs/1505.04597

3.	DeepLabV3+
   
Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV, pp. 801–818.

https://arxiv.org/abs/1802.02611

4.	PSPNet
   
Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network. CVPR, pp. 2881–2890.

https://arxiv.org/abs/1612.01105
________________________________________

### 2. Keras and Practical Implementation Examples

4.	Keras Team. (2024).
   
DeepLabV3+ Semantic Segmentation

Image segmentation with DeepLabV3+.

https://keras.io/examples/vision/deeplabv3_plus/

6.	Keras Team. (2023).
   
Fully Convolutional Network (FCN)

Image segmentation using a fully convolutional network.

https://keras.io/examples/vision/fully_convolutional_network/

7.	Keras Team. (2023).
   
U-Net on Oxford Pets

Image segmentation using U-Net.

https://keras.io/examples/vision/oxford_pets_image_segmentation/

8.	Keras Team. (2024).
   
Segment Anything with SAM

Integrating SAM for image segmentation.

https://keras.io/examples/vision/sam/
________________________________________

### 3. Remote Sensing and Semantic Segmentation (Satellite/UAV)
   
8.	Šćepanović, S., Antropov, O., Laurila, P., Rauste, Y., Ignatenko, V., & Praks, J. (2019).
   
Wide-Area Land Cover Mapping with Sentinel-1 Imagery using Deep Learning Semantic Segmentation Models. arXiv.

https://arxiv.org/abs/1912.05067

10.	Ulmas, P. & Liiv, I. (2020).
    
Segmentation of Satellite Imagery using U-Net Models for Land Cover Classification. arXiv.

https://arxiv.org/abs/2003.02899

11.	Zhang, G., Nur, S., Wang, C., & Quan, L. (2023).
An Improved Semantic Segmentation Algorithm for High-Resolution Remote Sensing Images Based on DeepLabV3+. Scientific Reports.
https://www.nature.com/articles/s41598-024-84795-1

### 4. Deep Learning & Computer Vision Textbooks
   
11.	Goodfellow, I., Bengio, Y., & Courville, A. (2016).

Deep Learning. MIT Press.

http://www.deeplearningbook.org

13.	Chollet, F. (2021).
    
Deep Learning with Python (2nd ed.). Manning Publications.

14.	Géron, A. (2022).
    
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media.

16.	Szeliski, R. (2022).
    
Computer Vision: Algorithms and Applications (2nd ed.). Springer.

https://szeliski.org/Book/
________________________________________
### References for Model Architecture Images

15.	PSPNet Diagram
    
Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network (PSPNet). GitHub Repository: https://github.com/hszhao/semseg.

Figure source: pspnet.png retrieved from

https://github.com/hszhao/semseg/blob/master/figure/pspnet.png

17.	DeepLabV3+ Keras Implementation
    
Keras Team. (2022). Image Segmentation with DeepLabV3+. Keras Examples.

Retrieved from: https://keras.io/examples/vision/deeplabv3_plus/

18.	U-Net Architecture Diagram
    
Milesial. (2020). PyTorch U-Net for Image Segmentation. GitHub Repository.

Retrieved from: https://github.com/milesial/Pytorch-UNet
________________________________________

### Image Dataset

19.	UAVid Dataset – Urban Semantic Segmentation

UAVid Dataset. (2018). A semantic segmentation dataset for urban scene understanding using UAV imagery. Retrieved from https://uavid.nl/

20.	Semantic Segmentation Dataset – Humans in the Loop
    
Humans in the Loop. (n.d.). Semantic Segmentation Dataset for Buildings, Roads, and Trees. Retrieved from https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset-2/

21.	Bhuvan Thematic Services – NRSC/ISRO
    
National Remote Sensing Centre (NRSC). (n.d.). Bhuvan Thematic Data: Land Use/Land Cover (LULC) and Urban Mapping. Indian Space Research Organisation (ISRO). Retrieved from https://bhuvan-app1.nrsc.gov.in/thematic/thematic/index.php



