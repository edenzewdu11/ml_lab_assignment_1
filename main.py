from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import uvicorn

# Load models
logistic_model = joblib.load("logistic_model_wine.pkl")
tree_model = joblib.load("decision_tree_model_wine.pkl")

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API is running"}

def get_quality_category(prob):
    if prob < 0.4:
        return "Poor", "🔴"
    elif prob < 0.6:
        return "Average", "🟡"
    else:
        return "Good", "🟢"

@app.post("/predict")
def predict(features: WineFeatures, model_type: str = Query("logistic")):
    input_data = [[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.ph,
        features.sulphates,
        features.alcohol
    ]]

    try:
        logit_proba = logistic_model.predict_proba(input_data)[0][1]

        if hasattr(tree_model, "predict_proba"):
            tree_proba = tree_model.predict_proba(input_data)[0][1]
        else:
            tree_pred = tree_model.predict(input_data)[0]
            tree_proba = 0.8 if tree_pred == 1 else 0.2

        if model_type.lower() == "logistic":
            final_proba = logit_proba + 0.05
        else:
            final_proba = tree_proba - 0.05

        final_proba = max(0.0, min(final_proba, 1.0))

        quality, emoji = get_quality_category(final_proba)

        return {
            "quality": quality,
            "emoji": emoji,
            "confidence": round(final_proba * 100, 2),
            "message": f"This wine is predicted to be of {quality.lower()} quality with {final_proba*100:.1f}% confidence."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔥 REQUIRED FOR RENDER 🔥
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
