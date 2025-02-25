import utils.utils as ut
from model.model_preprocessing import get_df_prediction, split_dataset, balance_dataset, apply_scaling
from model.model_utils import needs_scaling
from model.model_trainer import train_and_evaluate

def run_model(df):
    """
    Runs the process of training, evaluating, and applying the model on the provided dataset.

    This function performs multiple steps including loading the configuration, splitting the dataset into train,
    test, and validation sets, balancing the training data if required, applying scaling transformations,
    and finally training and evaluating the model. The configuration file dictates specific operations such
    as whether scaling is necessary and other model-related parameters.

    Arguments:
        df: DataFrame containing the input data for the model.

    Returns:
        None.

    Raises:
        None.
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