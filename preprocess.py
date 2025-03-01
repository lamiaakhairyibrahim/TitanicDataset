import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
class preprocessing:
    def __init__(self , data):
        print("build a predictive model that answers the question: “what sorts of people were more likely to survive?” ")
        self.data = pd.DataFrame(data)
        self.process()

    def process(self):
        print(f"show information about the dataset : {self.data.info()}" )
        print("first will remove the column that not useful, like ->")
        print("PassengerId , Name , Ticket")
        data_ = self.data.drop([ 'PassengerId' ,'Name' , 'Ticket'] , axis= 1)
        print(f"show information about the dataset : {data_ .info()}" )
        print("show number of missing value :")
        print(data_.isnull().sum())
        print("Since this column contains too many missing values, we will remove it completely.")
        data_ = data_.drop(['Cabin'] ,axis = 1)
        print("information about dataset after remove (Cabin) ")
        print(data_.info())
        print(data_.isnull().sum())
        print("We can fill the missing values in column (Age) with the mean.")
        data_['Age']= data_['Age'].fillna(data_['Age'].mean() )
        print(data_.info())
        print(data_.isnull().sum())
        print("We can fill the missing values in column (Embarked) with the mode.")
        data_['Embarked']= data_['Embarked'].fillna(data_['Embarked'].mode()[0] )
        print(data_.info())
        print(data_.isnull().sum())
        print("Displaying the unique values and their counts in column (Sex) ,(Embarked) , (Survived).")
        print("the unique values in (Sex) and its number :")
        print(data_['Sex'].unique())
        print(data_['Sex'].value_counts())
        print("the unique values in (Embarked) and its number :")
        print(data_['Embarked'].unique())
        print(data_['Embarked'].value_counts())
        print("the unique values in (Survived) and its number :")
        print(data_['Survived'].unique())
        print(data_['Survived'].value_counts())
        print("Convert categorical columns to numerical")
        print("Label Encoding for 'Sex' column")
        le = LabelEncoder()
        da = pd.DataFrame(le.fit_transform(data_['Sex']) , columns=['sex_edit'])
        #print(da)
        data_ = pd.concat([data_ , da ] , axis= 1)
        #print(data_.head())
        data_ = data_.drop(['Sex'] , axis= 1)
        print(data_.head())
        print("one-hot-encoding for 'Embarked' column")
        titanic_df = pd.get_dummies(data_, columns=["Embarked"], drop_first=True)
        print(titanic_df.head())
        print(titanic_df.info())
        sns.countplot(x='Survived', data=titanic_df, palette=['red', 'green'])
        plt.title("Survival Count")
        plt.show()
        sns.histplot(titanic_df, x="Age", hue="Survived", kde=True, bins=30, palette=['red', 'green'])
        plt.title("Age Distribution by Survival")
        plt.show()
        sns.barplot(x="sex_edit", y="Survived", data=titanic_df)
        plt.title("Survival Rate by Gender")
        plt.show()
        sns.barplot(x="Pclass", y="Survived", data=titanic_df)
        plt.title("Survival Rate by Ticket Class")
        plt.show()

        x = titanic_df.drop(['Survived'] , axis=1)
        y = titanic_df['Survived']

        return x , y




