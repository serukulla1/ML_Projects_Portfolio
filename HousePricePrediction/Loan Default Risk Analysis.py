# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:36:19 2025

@author: swapna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
%matplotlib inline

### setting up plot style

style.use('seaborn-poster')
style.use('fivethirtyeight')  #Apply fivethirtyeight visuals

##Supress warnings

import warnings
warnings.filterwarnings('ignore')


## Import the dataset

import os
for dirname, _, filenames in os.walk(r'C:\Swapna\Learning\PYTHON\FS DataScience - Theory\ML\18th - Regression Project\bank risk analysis'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
 #Reading the csv's
applicationDF = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience - Theory\ML\18th - Regression Project\bank risk analysis\csv data\application_data.csv')  
previousDF = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience - Theory\ML\18th - Regression Project\bank risk analysis\csv data\previous_application.csv')     
    

