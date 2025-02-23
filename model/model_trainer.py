import numpy as np
import utils.utils as ut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from model.model_utils import evaluate_model, best_threshold

def train_and_evaluate(x_train, x_test, x_val, y_train, y_test, y_val, df_prediction, config):
    """
    """
    model_name = config["applied_model"]["name"]
    print(f'{ut.get_time_now()} :: Running model: Training {model_name}')

    #Filter only numeric columns to avoid errors with timestamps
    x_train = x_train.select_dtypes(include=[np.number])
    x_test = x_test.select_dtypes(include=[np.number])
    x_val = x_val.select_dtypes(include=[np.number])

    #Model selection based on `model_config.json`.
    if model_name == "LogisticRegression":
        model = LogisticRegression()
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=config["RandomForestClassifier"]["n_estimators"],
            max_depth=config["RandomForestClassifier"]["max_depth"],
            random_state=config["RandomForestClassifier"]["random_state"]
        )
    elif model_name == "xgboost":
        model = XGBClassifier(random_state=42, reg_lambda=1, reg_alpha=0.5, gamma=1, max_depth=3, eval_metric=['aucpr'])
    else:
        raise ValueError(f"Model '{model_name}' not recognized in configuration.")

    #Training the model
    model.fit(x_train, y_train)

    #Calulate score train and test
    y_score_train = model.predict_proba(x_train)[:, 1]
    y_score_test = model.predict_proba(x_test)[:, 1]
    y_score_val = model.predict_proba(x_val)[:, 1]
    best_threshold_score = best_threshold(y_train, y_score_train)

    #Making predictions on test
    y_prediction_test = (y_score_test >= best_threshold_score).astype(int)
    y_prediction_train = (y_score_train >= best_threshold_score).astype(int)
    y_prediction_val = (y_score_val >= best_threshold_score).astype(int)

    evaluate_model(y_train,y_prediction_train,'Training')
    evaluate_model(y_val,y_prediction_val,'Validation')
    evaluate_model(y_test,y_prediction_test,'Testing')

    #Predict the next hour's target
    next_hour_prediction = model.predict(df_prediction)
    print(f'Prediction for the next hour: {"UP" if next_hour_prediction == 1 else "DOWN"}')

    return model

