# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model utilizes Random Forest Classifier. It is intended to predict whether an indvidual has greater than $50,000 of income. The model operates off of US Census data.

## Intended Use
This model is intended to be used to predict whether an individual has an income of greater that $50,000. It is primarily intended for demonstrating mastery in machine learning devops. 

## Training Data
census.csv file retrieve from the 1994 census dataset. This dataset contains 15 features and 26,049 entries. The features consist of: age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary. THe data was split for 80% training and 20% testing.

## Evaluation Data
The evaluation data is made up of the remaining 6,512 entries not included in the training data.

## Metrics
Precision: 0.7446 | Recall: 0.6276 | F1: 0.6811
This shows that the model had 74.46% of predicted positives end up being true positives, 62.76% of the actual positives were identified, and a harmonic mean of precision and recall of 0.6811 indicating a balanced performance.

## Ethical Considerations
Bias: The model may have inherited biases present in the US Census dataset that was utilized.
Data Privacy: Data is publicly available. Compliant with data regulations.

## Caveats and Recommendations
This model is intended for educational use to demonstrate machine learning workflows as well as deployment pipelines associated with machine learning devops. The data is pulled from 1994 so likely won't reflect current conditions.