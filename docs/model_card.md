---

# For reference on model card, see the spec: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

# Model Card for Traffic Signs Model in Yolo Format

# Model details

## Model description

Developed by Diaa Kotb (@diaakotb) on July 2024. Version 1. YOLOv8

- **License:** Apache 2.0 open source license.
- **Model Location:** The model can be accessed at [Kaggle - Traffic Signs Detection using YOLOv8](https://www.kaggle.com/code/diaakotb/traffic-signs-detection-using-yolov8#Evaluating-model).

---

## INTENDED USE

This model is designed for engaging applications, particularly in traffic sign detection. It can be beneficial for autonomous driving systems, enhancing navigation for visually impaired individuals, and various other innovative uses.

While it is particularly aimed at data scientists, it is important to note that this model is **not suitable** for detecting specific traffic signs themselves; rather, it identifies the type of sign based on its characteristics.

---

## FACTORS

### Environmental Factors

- **Lighting conditions**: As we are working with images, we must take into account that the model's performance may depend on lightness. It can work differently in poor lighting conditions, nighttime, or shadowed areas.
- **Weather conditions**: Cameras need to see clearly the signal the model is analyzing, so fog, rain, or snow could affect traffic sign detection.

### Camera Quality
- The model's performance may vary depending on the camera’s resolution or the model used to take the photographs. A higher-quality camera may lead to better results compared to a low-quality camera.

### Obstacles
- Traffic signs can be obscured or distorted (e.g., broken, painted over, or blocked by a tree), which can lead to detection errors.

---

## METRICS

The performance of the YOLOv8 model for traffic sign detection is measured using the following key metrics:

- **Precision**: Measures the proportion of true positive predictions out of all positive predictions. High precision means the model rarely misclassifies negative objects as positive. In this case, if the model detects a traffic sign, it’s very likely correct.
- **Recall**: Measures the proportion of actual positives correctly identified. High recall means that the model detects most traffic signs in an image, minimizing false negatives.
- **mAP50**: Mean average precision calculated at an IoU threshold of 0.5. It measures how well the predicted bounding box overlaps with the ground truth bounding box. High mAP50 indicates accurate and well-localized object detections.
- **mAP50-95**: A stricter version of mAP50, evaluating performance across a range of IoU thresholds (from 0.5 to 0.95). High scores reflect good performance even with stricter overlap requirements.

---

## EVALUATION DATA

The **YOLOv8 model** was evaluated using the **Traffic Signs Dataset in YOLO format**, which is derived from the **German Traffic Sign Detection Benchmark (GTDRB)**. 

- **Dataset Repository**: [Kaggle: Traffic Signs Dataset in YOLO Format](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/)
- **Dataset Paper**: [Stallkamp et al. (2012)](https://www.sciencedirect.com/science/article/abs/pii/S0893608012000457)

The dataset categorizes traffic signs into four groups:
- **Prohibitory** (e.g., speed limit, no overtaking)
- **Danger** (e.g., bends, slippery roads)
- **Mandatory** (e.g., go straight, keep right)
- **Other** (e.g., stop, no entry)

---

## QUANTITATIVE ANALYSIS

- **Precision (P)**: 99% overall, indicating few false positives.
- **Recall (R)**: 89.4% overall, showing high capability of identifying actual traffic signs.
- **mAP50**: 95.4%, representing the mean average precision at 0.5 IoU.
- **mAP50-95**: 79.8%, reflecting precision across various IoU thresholds (0.5 to 0.95).

**Performance by category**:

- **Prohibitory Signs**:  
  Precision: 99.9%, Recall: 94%, mAP50: 98.9%, mAP50-95: 84%
- **Danger Signs**:  
  Precision: 100%, Recall: 98.8%, mAP50: 99.5%, mAP50-95: 87.1%
- **Mandatory Signs**:  
  Precision: 97.4%, Recall: 85.7%, mAP50: 92.4%, mAP50-95: 75.3%
- **Other Signs**:  
  Precision: 98.5%, Recall: 78.9%, mAP50: 90.9%, mAP50-95: 72.9%

The average inference speed is 12.3ms per image, making it suitable for real-time applications.

---

## ETHICAL CONSIDERATIONS

### Safety
The model is designed for use in contexts that could affect human safety, such as autonomous vehicles. Both false positives and false negatives could be dangerous. For instance, incorrectly detecting a “stop” sign in the middle of a highway could be catastrophic, and missing a stop sign could also lead to serious consequences.

### Bias
For the model to be universal, it should account for differences in traffic signs across countries and regions. Training the model on diverse traffic sign datasets from different countries is necessary to ensure it works across various environments.

### Privacy
The dataset includes images captured from public roads, potentially containing personal information like license plates or faces. It's important to handle this data responsibly to avoid privacy violations.

---

## CAVEATS AND RECOMMENDATIONS

### Limitations due to the training set
If the model is deployed in countries with significantly different traffic signs, additional training on local datasets is recommended.

### Model Updates
Regular updates to the model are necessary to handle new traffic sign designs or variations, especially as road systems evolve over time.

---

## CITATION

If you use this dataset, please cite the following paper:

**APA:**
Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). **The German Traffic Sign Recognition Benchmark: A multi-class classification competition**. _Neural Networks_, 32, 136-153. https://doi.org/10.1016/j.neunet.2012.02.016
