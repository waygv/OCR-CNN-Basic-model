# Handwriting Recognition Model

📜 **Overview**

This project is focused on building a Handwriting Recognition Model using Convolutional Neural Networks (CNN) with Connectionist Temporal Classification (CTC) loss for sequence prediction. The model is trained on a Kaggle dataset containing handwritten names.

## 📦 Dependencies

To run the project, you'll need the following libraries:

- [**TensorFlow**](https://www.tensorflow.org/) 🧠: For deep learning and training models.
- [**Keras**](https://keras.io/) 🛠️: For building and compiling the model.
- [**OpenCV**](https://opencv.org/) 🖼️: For image processing.
- [**Pandas**](https://pandas.pydata.org/) 📊: For data manipulation.
- [**Matplotlib**](https://matplotlib.org/) 📉: For visualizations.
- [**NumPy**](https://numpy.org/) 🔢: For numerical operations.

## 💾 Dataset

You can download the dataset from Kaggle's Handwriting Recognition dataset [here](https://www.kaggle.com/datasets/landlord/handwriting-recognition).

### Files in the dataset:
- **`written_name_train_v2.csv`**: Contains training data (image file names and associated labels).
- **`written_name_validation_v2.csv`**: Contains validation data.
- **`written_name_test_v2.csv`**: Contains test data.

### Dataset structure:
The dataset should be unzipped and placed in the following directories:
- **`train_v2/train/`**: Contains the training images.
- **`validation_v2/validation/`**: Contains the validation images.
- **`test_v2/test/`**: Contains the test images.

## 🧩 Model Architecture

The model is a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), trained using **Connectionist Temporal Classification (CTC)** loss. 

### Architecture Breakdown:
1. **Convolutional Layers** 🖼️: Extract features from the input images.
2. **MaxPooling Layers** 📉: Down-sample feature maps.
3. **Reshape Layer** 🔄: Adjust the data to fit the sequence-based model.
4. **Dense Layer** ⚙️: Fully connected layers to map the features to character classes.
5. **CTC Loss Layer** 💥: Apply CTC loss for sequence prediction.

For more on CNNs, you can read the [Convolutional Neural Networks guide](https://en.wikipedia.org/wiki/Convolutional_neural_network) 🔗.

## ⚙️ Running the Model

You can run the code as Jupyter Notebooks or as Python scripts based on your preference.

### To Run the Code:
1. Clone this repository or download the notebook.
2. Install the dependencies using pip:
   ```bash
   pip install tensorflow keras opencv-python pandas matplotlib numpy
