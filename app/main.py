import os
import re
import torch
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

trained_models = joblib.load(
    os.path.join(MODEL_DIR, "traditional_models.pkl")
)

config = joblib.load(
    os.path.join(MODEL_DIR, "llm_config.pkl")
)

churn_examples = config["churn_examples"]
loan_examples = config["loan_examples"]

churn_features = config["churn_features"]
loan_features = config["loan_features"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto"
)

model.eval()


def create_classification_prompt(record, task):
    features = "\n".join(
        f"{key}: {value}" for key, value in record.items()
    )

    if task == "Churn":
        instructions = """
You are a customer churn classification system.

Your task is to predict whether a customer will leave their
telecommunications service.

Possible labels:
- Yes: The customer will churn.
- No: The customer will not churn.

Analyse the customer information and choose the most likely label.
"""

    else:
        instructions = """
You are a loan approval classification system.

Your task is to predict the loan application outcome.

Possible labels:
- Approved: The loan application will be approved.
- Rejected: The loan application will be rejected.

Analyse the applicant information and choose the most likely label.
"""

    return f"""
{instructions}

Applicant/customer information:
{features}

Return only one label. Do not provide an explanation.
"""


def create_few_shot_prompt(record, task, examples):

    if task == "Churn":
        instructions = """
You are a customer churn classification system.
Predict whether a customer will leave their telecom service.
Possible labels: Yes or No.
"""
    else:
        instructions = """
You are a loan approval classification system.
Predict whether a loan application will be approved.
Possible labels: Approved or Rejected.
"""

    example_text = ""

    for i, example in enumerate(examples, start=1):
        features = "\n".join(
            f"{key}: {value}"
            for key, value in example["features"].items()
        )

        example_text += f"""
Example {i}:
{features}
Correct classification: {example['label']}
"""

    new_record = "\n".join(
        f"{key}: {value}" for key, value in record.items()
    )

    return f"""
{instructions}

Here are examples of previously classified records:

{example_text}

Now classify this new record:

{new_record}

Return only one label.
"""


def generate_predictions(prompts, task, batch_size=8):

    predictions = []

    for start in range(0, len(prompts), batch_size):

        batch_prompts = prompts[start:start + batch_size]

        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in batch_prompts
        ]

        formatted = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            for message in messages
        ]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated = outputs[
            :, inputs["input_ids"].shape[1]:
        ]

        responses = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        for response in responses:
            response = response.strip()

            if task == "Churn":
                match = re.search(r"\b(Yes|No)\b", response, re.I)
            else:
                match = re.search(
                    r"\b(Approved|Rejected)\b", response, re.I
                )

            predictions.append(
                match.group(1).title() if match else "Invalid"
            )

    return predictions


def predict_llm_batch(records, task, batch_size=8):
    prompts = [
        create_classification_prompt(record, task)
        for record in records
    ]
    return generate_predictions(prompts, task, batch_size)


def predict_few_shot(records, task, examples, batch_size=8):
    prompts = [
        create_few_shot_prompt(record, task, examples)
        for record in records
    ]
    return generate_predictions(prompts, task, batch_size)


app = FastAPI(
    title="LLM and Traditional ML Classification API",
    description="Customer churn and loan approval classification.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    model: str
    features: Dict[str, Any]


AVAILABLE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Qwen Zero-shot",
    "Qwen Few-shot"
]


DATASET_CONFIG = {
    "Churn": {
        "task": "Churn",
        "features": churn_features,
        "examples": churn_examples
    },
    "Loan Approval": {
        "task": "Loan Approval",
        "features": loan_features,
        "examples": loan_examples
    }
}


@app.get("/")
def home():
    return {
        "message": "Classification API is running",
        "available_datasets": list(DATASET_CONFIG.keys()),
        "available_models": AVAILABLE_MODELS
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def run_prediction(dataset, request):

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {AVAILABLE_MODELS}"
        )

    dataset_config = DATASET_CONFIG[dataset]
    expected_features = dataset_config["features"]
    supplied_features = request.features

    missing = set(expected_features) - set(supplied_features)
    extra = set(supplied_features) - set(expected_features)

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra)
            }
        )

    record = {
        feature: supplied_features[feature]
        for feature in expected_features
    }

    if request.model in ["Logistic Regression", "Random Forest"]:

        selected_model = trained_models[
            (dataset, request.model)
        ]

        input_df = pd.DataFrame([record])
        prediction = selected_model.predict(input_df)[0]

        probabilities = selected_model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

    elif request.model == "Qwen Zero-shot":

        prediction = predict_llm_batch(
            [record], dataset_config["task"]
        )[0]
        confidence = None

    else:

        prediction = predict_few_shot(
            [record],
            dataset_config["task"],
            dataset_config["examples"]
        )[0]
        confidence = None

    return {
        "dataset": dataset,
        "model": request.model,
        "prediction": str(prediction),
        "confidence": confidence
    }


@app.post("/predict/churn")
def predict_churn(request: PredictionRequest):
    return run_prediction("Churn", request)


@app.post("/predict/loan")
def predict_loan(request: PredictionRequest):
    return run_prediction("Loan Approval", request)
