# OCR-CNN-Basic-model
Run Jupyter Notebook or Python Scripts
You can run the code as Jupyter notebooks or as Python scripts depending on your preference.

bash
Copy
Edit
jupyter notebook
Dependencies:
tensorflow (for deep learning and training models)
keras (for building and compiling the model)
opencv-python (for image processing)
pandas (for data manipulation)
matplotlib (for visualizations)
numpy (for numerical operations)
Dataset
The dataset used for training and validation comes from Kaggle's Handwriting Recognition dataset. You can download the dataset [here](https://www.kaggle.com/datasets/landlord/handwriting-recognition).

Files from the dataset:

written_name_train_v2.csv: Contains training data (image file names and associated labels).
written_name_validation_v2.csv: Contains validation data.
written_name_test_v2.csv: Contains test data.
The dataset should be unzipped in the specified directories in the code:

train_v2/train/: Contains the training images.
validation_v2/validation/: Contains the validation images.
test_v2/test/: Contains the test images.
Model Architecture
The model is a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) trained using Connectionist Temporal Classification (CTC) loss. The model is defined as follows:

Convolutional Layers: Extract features from the input images.
MaxPooling Layers: Down-sample feature maps.
Reshape Layer: Adjust the data to fit the sequence-based model.
Dense Layer: Apply fully connected layers to map to character classes.
CTC Loss Layer: Apply CTC loss for sequence prediction.
