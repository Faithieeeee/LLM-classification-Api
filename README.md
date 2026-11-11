# LLM-Based Classification API

## Project Overview

This project explores the use of a pretrained Large Language Model (LLM)
for tabular classification tasks and compares its performance with
traditional machine learning algorithms.

The experiments focus on two classification problems:

- Customer Churn Prediction
- Loan Approval Prediction

The project evaluates whether a pretrained instruction-tuned LLM can
perform classification tasks commonly handled using traditional
machine learning techniques.

## Objectives

- Build traditional machine learning classification models.
- Implement an LLM-based classification approach using a pretrained model.
- Evaluate zero-shot and few-shot prompting strategies.
- Compare the performance of the different approaches.
- Expose the classifiers through a REST API.

## Datasets

### Customer Churn
Source: https://www.kaggle.com/datasets/rashadrmammadov/customer-churn-dataset

Target variable: `Churn`

### Loan Approval
Source: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

Target variable: `loan_status`

## Models Implemented

### Traditional Machine Learning
- Logistic Regression
- Random Forest Classifier

### Large Language Model
- Qwen2.5-1.5B-Instruct
- Zero-shot classification
- Few-shot classification

The Qwen model was used in its pretrained form. It was not trained
or fine-tuned from scratch.

## Methodology

The datasets were cleaned and prepared before model development.
Identifier columns were excluded from the features.

The traditional models used preprocessing pipelines involving
numerical imputation, categorical imputation, feature scaling,
and one-hot encoding where appropriate.

The data was divided into training and testing sets using an
80:20 split.

The LLM was evaluated using zero-shot and few-shot prompting.
Few-shot examples were selected from the training data only.

For a fair comparison, all four approaches were evaluated on the
same 100-record test subset for each dataset.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Precision, recall and F1-score in the main comparison use the
positive class: Yes for churn and Approved for loan approval.

## Results

The Random Forest classifier achieved the strongest performance
on the loan approval test subset, with approximately 98% accuracy.

The LLM showed weaker overall classification performance.
Its predictions also demonstrated class bias, particularly on
the customer churn dataset.

Few-shot prompting did not improve accuracy over zero-shot
prompting in these experiments.

The results show that the effectiveness of an LLM for tabular
classification depends on the dataset and prompting strategy.

## API

The project uses FastAPI to expose the classification models.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API homepage |
| GET | /health | Health check |
| POST | /predict/churn | Customer churn prediction |
| POST | /predict/loan | Loan approval prediction |

### Available Models

- Logistic Regression
- Random Forest
- Qwen Zero-shot
- Qwen Few-shot

Interactive API documentation is available at `/docs` when
the application is running.

## Project Structure

```text
llm-classification-api/
├── app/
│   └── main.py
├── notebooks/
│   └── classification_project.ipynb
├── models/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- LLM predictions were evaluated on a limited subset of 100 records per dataset.
- The pretrained LLM was not fine-tuned.
- The API currently requires locally available model artifacts.
- The Colab-hosted API is temporary and is not a permanent deployment.

## Future Improvements

- Evaluate larger language models.
- Explore fine-tuning approaches.
- Test additional tabular datasets.
- Improve prediction consistency and class balance.
- Deploy the API to a permanent hosting platform.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Qwen2.5-1.5B-Instruct
- FastAPI
- Google Colab

## Author

Faith Osemeke
