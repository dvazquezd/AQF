import utils.utils as ut
from model.model_preprocessing import get_df_prediction, split_dataset, balance_dataset, apply_scaling
from model.model_utils import needs_scaling
from model.model_trainer import train_and_evaluate

def run_model(df):
    """
    Executes the full pipeline of pre-processing, modeling, and evaluation on the given dataset. The function runs through
    a series of steps including dataset splitting, balancing, scaling, and model training/evaluation, governed by
    configurations provided through a model configuration file.

    Args:
        df: Input pandas DataFrame that contains input features required for model training and predictions.

    Returns:
        tuple: A tuple containing the training feature set (x_train), test feature set (x_test), training labels (y_train),
        and test labels (y_test) after all preprocessing steps and balancing (if applicable).
    """
    print(f'{ut.get_time_now()} :: Running model: Starting running model')

    # Loading config
    config = ut.load_config('model_config')
    df_prediction = get_df_prediction(df)

    # Splitting train and test
    x_train, x_test, x_val, y_train, y_test, y_val = split_dataset(df)

    # Applying balance if it is necessary
    x_train, y_train = balance_dataset(x_train, y_train, config)

    if needs_scaling(x_train, config):
        x_train, x_test, x_val, df_prediction = apply_scaling(x_train, x_test, x_val, df_prediction, config)

    df_prediction = df_prediction.drop(columns=['datetime'])

    train_and_evaluate(x_train, x_test, x_val, y_train, y_test, y_val, df_prediction, config)