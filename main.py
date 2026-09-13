from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import re

app = FastAPI(title="Career / Job-Change Prediction API (Multi-Model)")

encoder = joblib.load("encoder.pkl")
with open("metadata.json") as f:
    meta = json.load(f)

models = {name: joblib.load(info["file"]) for name, info in meta["models"].items()}
categorical_columns = meta["categorical_columns"]
numerical_columns = meta["numerical_columns"]
feature_orders = meta["feature_orders"]
best_model_name = meta["best_model_name"]


def sanitize_columns(cols):
    clean = []
    for c in cols:
        c = c.replace("<", "lt").replace(">", "gt")
        c = re.sub(r"[\[\]{}():,]", "_", c)
        clean.append(c)
    return clean


class Candidate(BaseModel):
    city: str
    gender: str
    relevent_experience: str
    enrolled_university: str
    education_level: str
    major_discipline: str
    experience: str
    company_size: str
    company_type: str
    last_new_job: str
    city_development_index: float
    training_hours: float


def build_feature_sets(raw: pd.DataFrame):
    cat_encoded = pd.DataFrame(
        encoder.transform(raw[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
    )

    plain = pd.concat([cat_encoded, raw[numerical_columns]], axis=1)
    plain.columns = plain.columns.astype(str)
    plain = plain.reindex(columns=feature_orders["plain"], fill_value=0)

    exp_num = raw["experience"].replace({"<1": 0, ">20": 21, "Unknown": 0}).astype(float)
    fe_num = raw[numerical_columns].copy()
    fe_num["experience_to_training_ratio"] = exp_num / (raw["training_hours"] + 1)
    fe_num["has_relevant_degree"] = (raw["major_discipline"] == "STEM").astype(int)
    fe = pd.concat([cat_encoded, fe_num], axis=1)
    fe.columns = fe.columns.astype(str)
    fe = fe.reindex(columns=feature_orders["fe"], fill_value=0)

    xgb = plain.copy()
    xgb.columns = sanitize_columns(xgb.columns)
    xgb = xgb.reindex(columns=feature_orders["xgb"], fill_value=0)

    return {"plain": plain, "fe": fe, "xgb": xgb}


@app.get("/")
def root():
    return {"status": "API is running", "best_model": best_model_name, "models_loaded": list(models.keys())}


@app.get("/metrics")
def metrics():
    return {name: info["metrics"] for name, info in meta["models"].items()}


@app.post("/predict")
def predict(candidate: Candidate):
    raw = pd.DataFrame([candidate.dict()])
    feature_sets = build_feature_sets(raw)

    results = {}
    for name, info in meta["models"].items():
        model = models[name]
        X_input = feature_sets[info["feature_type"]]
        proba_change = float(model.predict_proba(X_input)[0][1])
        will_change = proba_change >= info["threshold"]
        results[name] = {
            "job_change_probability": round(proba_change, 4),
            "stay_probability": round(1 - proba_change, 4),
            "label": "Likely to Change Jobs" if will_change else "Likely to Stay",
        }

    return {
        "best_model": best_model_name,
        "best_result": results[best_model_name],
        "all_models": results,
    }
