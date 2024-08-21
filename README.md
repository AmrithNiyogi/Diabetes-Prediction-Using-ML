# Diabetes Prediction Using ML

This repository contains a Jupyter Notebook for predicting diabetes using machine learning techniques. The notebook implements a model that classifies individuals as diabetic or non-diabetic based on various health parameters.

## Overview

The project involves loading a dataset, preprocessing the data, building a machine learning model, and evaluating its performance. The key steps include:

1. **Data Loading**: The dataset is loaded from a CSV file (`diabetes.csv`).
2. **Data Preprocessing**: Standard scaling is applied to the features to normalize the data.
3. **Model Building**: A Support Vector Machine (SVM) model is used for classification.
4. **Model Evaluation**: The model's performance is assessed using accuracy metrics.

## Requirements

To run the notebook, you will need the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `jupyter`

You can install these libraries using `pip`:

```bash
pip install numpy pandas scikit-learn jupyter
```

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AmrithNiyogi/Diabetes-Prediction-Using-ML.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd Diabetes-Prediction-Using-ML
    ```
3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Diabetes\ Prediction.ipynb
    ```

4. **Follow the steps in the notebook** to load the data, preprocess it, train the model, and evaluate its performance.

## Dataset

The dataset used for this project is `diabetes.csv`, which contains the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: A function which scores likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) indicating if the patient is diabetic (1) or not (0)

## Results

The model's performance is evaluated using accuracy, with the results indicating the effectiveness of the SVM model in predicting diabetes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---