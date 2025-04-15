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
 - After training the Random Forest model, its performance was evaluated using the test dataset. Below are the results:
   - ## Classification Report:
   | Survived | Precision | Recall | F1-score | Support |
   |----------|-----------|--------|----------|---------|
   | 0        | 0.84      | 0.91   | 0.87     | 110     |
   | 1        | 0.83      | 0.72   | 0.78     | 69      |

## How to Run
1. Creation of virtual environments
```Bash
python -m venv <name of your environment>
```
2. activation of environment
```Bash
<name of your environment>\Scripts\activate
```
3. Change the directory inside to the environment
```Bash 
cd <name of your environment>
```
4. creat folder in this directory
```Bash 
md src
```
5. Change the directory inside to src
```Bash
cd src
```
6. Colne this repository:
```Bash
git clone <url of repo >
```
7. install the required dependencies:
```Bash 
pip install -r requirements.txt
```
8. Run the credit_card_fraud_detection.py script:
```Bash 
python main.py <path of dataset>
```