# ExtremeDropClassifier

## Project Overview

The ExtremeDropClassifier is designed to identify and classify extreme drops within time-series data. 
Using `training.csv` and `testing.csv`, the model aims to detect instances marked as `S`, indicating significant 
deviations or issues within the series. This classification challenge involves preprocessing the data, 
feature engineering, and applying machine learning models to accurately predict these extreme drops on unseen data.

## Data Description

- **Training and Testing Data**: Includes `EventTime`, `Measure`, and `Class` columns.
- **Target Class**: `Class` column with `S` indicating a deviation. Preprocessed to binary encoding for modeling.

## Discussion
### Data Preprocessing and Exploration

- Successful reading and initial exploration of CSV files highlighted the presence of missing values exclusively 
    within the `Class` column, primarily marked as `NaN`, indicating a significant class imbalance.
- Preprocessing steps included handling missing values, converting the `Class` column to binary encoding, 
 and transforming the `EventTime` column to datetime format for further analysis.

### Feature Engineering

- Introduced time-based features and rolling features with a window size of 5 to capture temporal patterns 
and trends within the data, enhancing the model's ability to identify significant drops.

### Model Selection and Evaluation

- **Model Evaluation**: Two models, `Random Forest` and `Gradient Boosting`, were evaluated based on their accuracy on the test set.
- **Results**:
  - The Random Forest model demonstrated superior performance with an accuracy of `94.70%` in the first run 
    and was identified as the best model. However, a subsequent evaluation showed a decrease in performance, 
    with a balanced accuracy score on the testing set of `62.5%`, highlighting challenges in generalizing to 
    detect rare events within the data effectively.
  - The Gradient Boosting model showed slightly lower performance compared to Random Forest in both evaluations, 
  underscoring the difficulty of the task given the extreme class imbalance.

### Conclusions and Future Work

- The `ExtremeDropClassifier` project underscores the complexity of classifying rare events in time-series data, 
    especially in the face of significant class imbalances.
- Future improvements could include exploring more sophisticated imbalance handling techniques, 
experimenting with different feature engineering strategies, and testing more advanced models or ensemble 
methods to enhance predictive performance.

## Usage

- To run the model: `poetry run python main.py` from the project's root directory.
- Ensure all dependencies are installed as per the `pyproject.toml` file in the project's repository.

## Dependencies

- Python 3.12
- Pandas for data manipulation.
- Scikit-learn for model training and evaluation.