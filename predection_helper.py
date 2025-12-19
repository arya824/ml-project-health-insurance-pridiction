import pandas as pd
from joblib import load

model_rest = load(r"artifacts\model_rest.joblib")
model_young = load(r"artifacts\model_young.joblib")

# These are dictionaries: {"scaler": <sklearn_scaler>, "cols_to_scale": [...]}
scaler_rest = load(r"artifacts\scaler_rest.joblib")
scaler_young = load(r"artifacts\scaler_young.joblib")


def calculate_normalized_risk_score(medical_history: str) -> float:
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0,
    }

    max_score = 8 + 6  # 14
    min_score = 0 + 0  # 0

    parts = medical_history.lower().split(" & ")

    disease1 = parts[0] if len(parts) > 0 else "none"
    disease2 = parts[1] if len(parts) > 1 else "none"

    d1_score = risk_scores.get(disease1, 0)
    d2_score = risk_scores.get(disease2, 0)

    total_risk_score = d1_score + d2_score

    if max_score == min_score:
        return 0.0

    normalized_risk_score = total_risk_score / (max_score - min_score)
    return normalized_risk_score


def preprocess_input(input_dict):
    # IMPORTANT: must match the training columns of your XGBoost models
    expected_columns = [
        "age", "number_of_dependants", "income_lakhs", "insurance_plan",
        "genetical_risk",
        "gender_Male",
        "region_Northwest", "region_Southeast", "region_Southwest",
        "marital_status_Unmarried",
        "bmi_category_Obesity", "bmi_category_Overweight",
        "bmi_category_Underweight",
        "smoking_status_No Smoking", "smoking_status_Not Smoking",
        "smoking_status_Occasional", "smoking_status_Regular",
        "smoking_status_Smoking=0",
        "employment_status_Salaried", "employment_status_Self-Employed",
        # Do NOT add normalized_risk_score here; model was not trained with it
    ]

    insurance_plan_encoding = {"Bronze": 1, "Silver": 2, "Gold": 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        # Gender
        if key == "Gender" and value == "Male":
            df.loc[0, "gender_Male"] = 1

        # Region
        elif key == "Region":
            if value == "Northwest":
                df.loc[0, "region_Northwest"] = 1
            elif value == "Southeast":
                df.loc[0, "region_Southeast"] = 1
            elif value == "Southwest":
                df.loc[0, "region_Southwest"] = 1

        # Marital status
        elif key == "Marital Status" and value == "Unmarried":
            df.loc[0, "marital_status_Unmarried"] = 1

        # BMI category
        elif key == "BMI Category":
            if value == "Obesity":
                df.loc[0, "bmi_category_Obesity"] = 1
            elif value == "Overweight":
                df.loc[0, "bmi_category_Overweight"] = 1
            elif value == "Underweight":
                df.loc[0, "bmi_category_Underweight"] = 1

        # Smoking status
        elif key == "Smoking Status":
            if value == "Occasional":
                df.loc[0, "smoking_status_Occasional"] = 1
            elif value == "No Smoking":
                df.loc[0, "smoking_status_No Smoking"] = 1
            elif value == "Regular":
                df.loc[0, "smoking_status_Regular"] = 1
            # Only add more encodings if these existed during training

        # Employment status
        elif key == "Employment Status":
            if value == "Salaried":
                df.loc[0, "employment_status_Salaried"] = 1
            elif value == "Self-Employed":
                df.loc[0, "employment_status_Self-Employed"] = 1

        # Insurance plan
        elif key == "Insurance Plan":
            df.loc[0, "insurance_plan"] = insurance_plan_encoding.get(value, 1)

        # Numeric features
        elif key == "Age":
            df.loc[0, "age"] = value
        elif key == "Number of Dependants":
            df.loc[0, "number_of_dependants"] = value
        elif key == "Income in Lakhs":
            df.loc[0, "income_lakhs"] = value
        elif key == "Genetical Risk":
            df.loc[0, "genetical_risk"] = value

    # Optional: compute risk score for display only (not used by model)
    _ = calculate_normalized_risk_score(input_dict["Medical History"])

    df = handle_scaling(input_dict["Age"], df)
    return df


def handle_scaling(age, df):
    # scaler_young / scaler_rest are dicts: {"scaler": ..., "cols_to_scale": [...]}
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    # If your scaler was fitted with an extra temp column, keep this;
    # otherwise you can remove income_level-related lines.
    df.loc[0, "income_level"] = None

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    if "income_level" in df.columns:
        df.drop("income_level", axis=1, inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict["Age"] <= 25:
        prediction = model_rest.predict(input_df)
    else:
        prediction = model_young.predict(input_df)

    return int(prediction)
