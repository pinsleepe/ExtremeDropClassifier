import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Example classifier
from loguru import logger


class ModelSelection:
    def __init__(self, X, y):
        """
        Initializes the ModelSelection class with the dataset.

        :param X: A DataFrame or numpy array of shape [n_samples, n_features] containing the feature data.
        :param y: A numpy array or list containing the target values.
        """
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.X = X
        self.y = y
        self.model = None  # Placeholder for the selected model
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = ""

        # Mapping of model names to classifier objects
        # model parameters adjusted for imbalanced dataset
        self.available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100,
                                                    class_weight='balanced',
                                                    max_depth=None,
                                                    min_samples_split=2,
                                                    min_samples_leaf=1,
                                                    max_features='sqrt'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100,
                                                            learning_rate=0.1,
                                                            max_depth=3,
                                                            min_samples_split=2,
                                                            min_samples_leaf=1,
                                                            subsample=0.8,
                                                            max_features='sqrt'),
        }

    def split_data(self, test_size=0.2, random_state=None):
        """
        Splits the data into training and testing sets.

        :param test_size: Fraction of the dataset to be used as test set.
        :param random_state: Seed used by the random number generator.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        logger.info("Data split into training and testing sets.")

    def select_model(self, model_name='random_forest'):
        """
        Selects the machine learning model for the task.

        :param model_name: An sklearn classifier instance. Defaults to 'random_forest'.
        """
        try:
            self.model = self.available_models[model_name]
            logger.info(f"Model selected: {model_name}")
        except KeyError:
            logger.error(
                f"Model {model_name} not recognized. Available models are: {list(self.available_models.keys())}")
            return

    def fit_model(self, model):
        """
        Fits the model on the training data.
        """

        model.fit(self.X_train, self.y_train)
        logger.info("Model fitted on the training data.")

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)
        accuracy, _ = self.evaluation_report(predictions, self.y_test)
        return accuracy

    # TODO: Add target name to parameter
    def evaluation_report(self, predictions, y_test):
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=['Normal', 'Issue'])

        logger.info(f"Accuracy on test set: {accuracy}")
        logger.info("Classification Report:\n" + report)

        return accuracy, report

    def find_best_model(self):
        for name, model in self.available_models.items():
            logger.info(f"Evaluating model: {name}")
            self.fit_model(model)
            accuracy = self.evaluate_model(model)
            logger.info(f"Accuracy for {name}: {accuracy}")

            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name

        logger.info(f"Best model: {self.best_model_name} with accuracy: {self.best_score}")

    def save_best_model(self, filename):
        """
        Saves the best model as a pickle file.

        :param filename: The path and name of the file where to save the model.
        """
        if self.best_model is not None:
            with open(filename, 'wb') as file:
                pickle.dump(self.best_model, file)
            logger.info(f"Best model saved to {filename}")
        else:
            logger.error("No best model to save.")

    # TODO: Add pickle handling to FileHandler
    def predict_with_best_model(self, X_new, model_filename):
        """
        Loads the best model from a pickle file and uses it to make predictions on a new dataset.

        :param X_new: The new dataset to predict on. Ensure it's preprocessed similarly to the training data.
        :param model_filename: The path and name of the file where the best model is saved.
        :return: The predictions made by the loaded model, or None if an error occurs.
        """
        try:
            # Load the model from the pickle file
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)

            # Make predictions with the loaded model
            predictions = model.predict(X_new)
            return predictions
        except Exception as e:
            logger.error(f"Failed to load the model and predict. Error: {e}")
            return None
