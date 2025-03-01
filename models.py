from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix , classification_report , ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class model:
    def __init__(self , x , y):
        self.x = x
        self.y = y
        self.split_data()

    def split_data(self ):
        self.x_train , self.x_test , self.y_train , self.y_test =  train_test_split(self.x , self.y , test_size= 0.2 , random_state= 0)

    def randam(self):
        self.model = RandomForestClassifier(class_weight='balanced')
        self.model.fit( self.x_train , self.y_train )
        self.test_pred()

        
    def test_pred(self):
        self.y_pred = self.model.predict(self.x_test)
        conf_matr = confusion_matrix(self.y_test , self.y_pred)
        print(conf_matr)
        conf_matr_report = classification_report(self.y_test ,self.y_pred)
        print(conf_matr_report)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr, display_labels=['Died', 'Survived'])
        disp.plot(cmap="Blues")
        plt.show()