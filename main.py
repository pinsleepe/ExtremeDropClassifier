from pathlib import Path
from extreme_drop_classifier.file_handler import FileHandler
from extreme_drop_classifier.data_exploration import DataExploration
from extreme_drop_classifier.data_preprocessing import DataPreprocessing
from extreme_drop_classifier.feature_engineering import FeatureEngineering
from extreme_drop_classifier.model_selection import ModelSelection
from loguru import logger


def load_dataset(file_path):
    file_handler = FileHandler(file_path)
    return file_handler.read_csv()


def preprocess_dataset(df):
    preprocessor = DataPreprocessing()
    return preprocessor.transform(df)


def engineer_features(df):
    feature_engineer = FeatureEngineering()
    df = feature_engineer.add_time_based_features(df)
    df = feature_engineer.add_rolling_features(df, window_size=5)
    return df


def prepare_features(df, feature_columns):
    X = df[feature_columns]
    y = df['Class']
    return X, y


def process_and_select_model(training_df, feature_columns):
    preprocessed_df = preprocess_dataset(training_df)
    engineered_df = engineer_features(preprocessed_df)
    X, y = prepare_features(engineered_df, feature_columns)

    model_selector = ModelSelection(X, y)
    model_selector.split_data(test_size=0.2, random_state=42)
    model_selector.find_best_model()

    return model_selector


def save_model(model_selector, model_path):
    model_selector.save_best_model(filename=model_path)


def predict_and_evaluate(model_selector, testing_df, model_path):
    # Exclude 'Class' column from features used for prediction
    features_for_prediction = testing_df.drop(columns=['Class'])

    predictions = model_selector.predict_with_best_model(features_for_prediction, model_path)

    if predictions is not None:
        # Assuming 'Class' column exists in testing_df for comparison
        model_selector.evaluation_report(predictions, testing_df['Class'])
    else:
        logger.error("Prediction failed. No predictions to evaluate.")


def main():
    base_path = Path(__file__).parent
    training_file_path = base_path / 'data' / 'training.csv'
    testing_file_path = base_path / 'data' / 'testing.csv'

    training_df = load_dataset(training_file_path.as_posix())
    testing_df = load_dataset(testing_file_path.as_posix())

    # TODO: Read feature columns from config file
    # Feature columns to be used for modeling
    feature_columns = ['Measure', 'hour', 'dayofweek', 'rolling_mean_5', 'rolling_std_5']

    # Process training data and select the best model
    model_selector = process_and_select_model(training_df, feature_columns)

    # Save the best model
    model_path = base_path / 'models' / 'best_model.pkl'
    save_model(model_selector, model_path.as_posix())

    # Preprocess and engineer features on testing data
    testing_df_preprocessed = preprocess_dataset(testing_df)
    testing_df_engineered = engineer_features(testing_df_preprocessed)
    testing_df_engineered = testing_df_engineered[feature_columns + ['Class']]

    # Predict and evaluate on testing data
    predict_and_evaluate(model_selector, testing_df_engineered, model_path.as_posix())


if __name__ == '__main__':
    main()
