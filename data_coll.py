from src.preprocess import preprocessing
import pandas as pd
from src.models import model

class Collect:
        def __init__(self , path):
            self.path = path
            self.fan()
    
        def fan(self):
            data = pd.read_csv(self.path)
            data_= preprocessing(data)    
            feature , label = data_.process()
            print(feature.head())                        
            print(label.head())   
            m = model(feature , label)      
            m.randam()   
