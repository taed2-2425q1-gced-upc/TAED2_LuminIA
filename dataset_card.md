---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{}
---
# Dataset Card for Traffic Signs Dataset in YOLO format

The **Traffic Signs Dataset in YOLO format** provides images and YOLO-style annotations for traffic sign detection, derived from the German Traffic Sign Detection Benchmark. It categorizes signs into four groups: Prohibitory, Danger, Mandatory, and Other.

## Dataset Details

### Dataset Description

This **Traffic Signs Dataset in YOLO format** is designed for object detection tasks, specifically for detecting traffic signs. It includes labeled images in JPG format along with corresponding text files containing annotations of bounding boxes in YOLO format. These annotations describe the class number and the bounding box dimensions (center coordinates, width, and height).

The dataset is originated from the German Traffic Sign Detection Benchmark (GTDRB) and categorizes traffic signs into four groups:
- Prohibitory (e.g., speed limit, no overtaking)
- Danger (e.g., bends, slippery roads)
- Mandatory (e.g., go straight, keep right) 
- Other (e.g., stop, no entry)

- **Curated by:** Valentyn Sichkar
- **Funded by:** J. Stallkamp
- **License:** The licensing status of the dataset hinges on the legal status of the dataset published in Kaggle, which is unclear

### Dataset Sources

- **Repository:** https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/
- **Paper (GTDRB):** https://www.sciencedirect.com/science/article/abs/pii/S0893608012000457

## Uses

The **Traffic Signs Dataset in YOLO format** is versatile and can be used across various domains involving traffic sign recognition and object detection. It is intended to be used for:
- **Training machine learning models** for traffic sign detection and classification in diverse contexts such as autonomous driving, smart traffic systems, and roadside monitoring.
- **Benchmarking and validating object detection algorithms** by offering a standardized dataset with well-labeled images and bounding boxes, allowing researchers to compare performance metrics like accuracy and efficiency across models.
- **Enhancing computer vision research and development** by providing a robust dataset for improving traffic sign recognition, especially in areas where European-style traffic signs are prevalent.
- **Developing AI systems** for real-time traffic monitoring and safety enforcement, enabling automated detection of traffic violations or road hazards.

### Direct Use

The **Traffic Signs Dataset in YOLO format** can be applied directly for specific, technical purposes, leveraging its ready-to-use structure:
- **Training YOLO-based object detection models** without modification, utilizing pre-labeled images and annotations formatted specifically for YOLO models, optimizing the dataset for fast object detection tasks.
- **Developing real-time applications** such as autonomous vehicle systems, where fast and accurate detection of traffic signs is crucial, utilizing the dataset's YOLO format for immediate integration into production environments.
- **Fine-tuning pre-trained YOLO models** by adapting the dataset to models trained in similar environments, particularly in regions with European traffic signs, to improve accuracy in traffic sign detection tasks.
- **Validating model performance** by leveraging built-in metrics like mean average precision (mAP) to evaluate detection efficiency and accuracy during training and testing phases.

### Out-of-Scope Use

The **Traffic Signs Dataset in YOLO format** is intended for object detection tasks, and certain uses may be considered out-of-scope or inappropriate. Misuse or malicious use could include:
- **Non-traffic-related object detection tasks**: Using the dataset to train models for detecting objects unrelated to traffic signs (e.g., animals, human faces, or household items) would not yield meaningful results, as the dataset is specifically designed for traffic sign detection.
- **Manipulating or spoofing traffic systems**: Using the dataset to develop systems or AI models that could deliberately alter or misinterpret traffic sign detection in real-world applications, potentially causing safety hazards or violations.
- **Biased or discriminatory applications**: Applying the dataset in regions or environments where the traffic signs differ significantly from those included (e.g., using European traffic signs in countries with vastly different signage systems) could lead to biased or ineffective detection systems that misidentify signs, putting users at risk.

## Dataset Structure

The **Traffic Signs Dataset in YOLO format** is organized as follows:
- **Images**: The dataset consists of JPG images of traffic signs. Each image is a separate file and is named to correspond with its annotation file.
- **Annotations**: For each image, there is a corresponding text file in YOLO format that contains the annotations. Each line in the text file represents one traffic sign and follows the format: `class_number center_x center_y width height`. The coordinates are normalized to the dimensions of the image.
- **Class Labels**: The dataset includes a predefined list of class labels representing different traffic sign categories. Each class is assigned a unique number:
    - **Prohibitory**: 0
    - **Danger**: 1
    - **Mandatory**: 2
    - **Other**: 3
- **Data Splits**: The dataset may be split into training, validation, and test sets. For instance, a common approach is to use 70% of the data for training, 15% for validation, and 15% for testing. The splits are designed to ensure that the model can generalize well across different sets of traffic signs.
- **File Naming Convention**: Images and annotations are paired by naming convention. Each image file (e.g., `img001.jpg`) has a corresponding annotation file (e.g., `img001.txt`). The annotation file contains the bounding box coordinates and class labels in YOLO format.
- **Image Dimensions**: The dimensions of the images are consistent within the dataset.
- **Metadata**: Any additional metadata associated with the images (e.g., capture conditions, location, or time) is not included in this dataset.

## Dataset Creation

### Curation Rationale

The **Traffic Signs Dataset in YOLO format** was created to support the development and evaluation of object detection models specifically for traffic signs. The primary motivation behind the dataset is to provide a standardized, high-quality resource that can be used to train and test machine learning models in the context of autonomous driving and traffic monitoring systems. By using data derived from the German Traffic Sign Detection Benchmark, the dataset aims to facilitate research and advancements in traffic sign recognition, which is crucial for enhancing road safety and enabling intelligent transportation systems.

### Source Data

#### Data Collection and Processing

The source data for the **Traffic Signs Dataset in YOLO format** originates from the German Traffic Sign Detection Benchmark (GTDRB). The data collection process involves capturing images of various traffic signs in different conditions to create a diverse and representative dataset. The processing steps include:
- **Data Selection Criteria**: Images are selected to ensure a comprehensive coverage of traffic sign categories and conditions. The dataset includes a variety of traffic signs to enhance the robustness of object detection models.
- **Filtering**: Images are filtered to remove those that are not clearly visible or do not contain relevant traffic signs. This ensures that the dataset focuses on high-quality, useful examples.
- **Normalization**: The annotations are formatted to fit the YOLO object detection framework, which includes normalizing bounding box coordinates to the image dimensions.
- **Tools and Libraries**: Tools and libraries used for processing include image editing software for cropping and annotating images, as well as scripts for converting annotations into YOLO format.

#### Who are the source data producers?

The source data was originally created by J. Stallkamp and other researchers and contributors to the German Traffic Sign Detection Benchmark (GTDRB) in the field of computer vision and machine learning.

### Annotations 

This dataset does not contain any additional annotations.

## Bias, Risks, and Limitations

The **Traffic Signs Dataset in YOLO format** has several potential biases and limitations that users should be aware of:
- **Geographic Bias**: The dataset is derived from the German Traffic Sign Detection Benchmark, which means it primarily features traffic signs used in Germany. This may lead to geographic bias when the dataset is used in regions with different traffic sign designs, potentially affecting the model's performance in those areas.
- **Environmental Conditions**: The dataset may not cover all possible environmental conditions under which traffic signs appear. Variations in weather, lighting, or signage wear-and-tear may not be fully represented, which could limit the robustness of models trained on this dataset.
- **Cultural Bias**: The dataset reflects European traffic signs and might not be applicable to regions with significantly different traffic signage conventions. This could limit the dataset's utility for training models intended for global applications.

### Recommendations

To address the bias, risks, and limitations of the dataset, consider the following recommendations:
- **Supplement with Diverse Data**: To mitigate geographic and cultural biases, consider supplementing the dataset with traffic sign data from other regions or countries. This will help improve the model's performance across different traffic sign designs.
- **Enhance Environmental Coverage**: Include data from a variety of environmental conditions such as different lighting, weather conditions, and sign states (e.g., damaged or obscured signs) to improve model robustness.
- **Adapt Models for Regional Variations**: When deploying models in new regions, fine-tune them using locally relevant traffic sign data to ensure accurate detection and classification.

## Citation

**APA:**
Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). **The German Traffic Sign Recognition Benchmark: A multi-class classification competition**. _Neural Networks_, 32, 136-153. https://doi.org/10.1016/j.neunet.2012.02.016