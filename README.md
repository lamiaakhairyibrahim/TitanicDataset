# Titanic classification 
## Table of Contents

- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run](#how-to-run)


## Dataset
[dataset in kaggel](https://www.kaggle.com/competitions/titanic/data)
## Project Workflow
- The data preprocessing step is handled through a dedicated preprocessing class, which automates the preparation of the Titanic dataset to answer the question:
"What sorts of people were more likely to survive?"
1. *Dataset Overview*

    - Displays basic information and missing values in the dataset.

    - Removes uninformative columns such as PassengerId, Name, and Ticket.

2. *Handling Missing Values*

   - Drops the Cabin column due to a high percentage of missing data.

   - Fills missing values in the Age column with the mean.

   - Fills missing values in the Embarked column with the mode.

3. *Exploratory Analysis*

   - Displays unique values and distributions in categorical columns (Sex, Embarked, Survived).

   - Visualizes survival counts and relationships between survival and features like Age, Sex, and Pclass.

4. *Encoding Categorical Variables*

   - Applies Label Encoding to the Sex column.
   - Applies One-Hot Encoding to the Embarked column, dropping the first category to avoid multicollinearity.

5. *Output*

     - Returns the cleaned and preprocessed features (X) and target (y) for model training.

6. *Model Building and Evaluation*
   - The model class is responsible for building and evaluating a predictive model using a Random Forest Classifier. The process involves:

   - *Data Splitting:*
      -  Splits the preprocessed dataset into training and testing sets using an 80/20 split.

   - Model Training:
      - Trains a RandomForestClassifier with class_weight='balanced' to account for any imbalance in the survival classes.

    - Model Evaluation:
       - After training, the model makes predictions on the test set and evaluates performance using:

        - Confusion Matrix

        - Classification Report (Precision, Recall, F1-score, Accuracy)

    - Visual Display of the confusion matrix for better interpretability.
## Results
## How to Run