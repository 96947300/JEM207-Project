# Analysis
The next steps constitude the treatment and analysis of the data. Additionally, the establishment of the best model.

## Importing the dataset:
For this we use the deafult method for importing data from UCI repository.

```py linenums="1" title="Importing UCI dataset" hl_lines="7 8"
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
features = heart_disease.data.features 
targets = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 
```
By default, the data correspond to a sample of adults from Cleveland. It is constituted by 303 observations, 13 explanatory variables, and 1 multicategorical target variable.

## Pre-setting and handling missing data:
After exploring a bit the dataset and determining the existence of missing data we opt to do two processes. The first one was the assign the correct data types according to the values of each attribute, and the second imputing and dropping rows with missing data.

```py linenums="1" title="Assigning correct data types"
#Setting of correct data types
base_df = base_df.astype({"sex": "category", "cp": "category", "fbs": "category", "restecg": "category",
               "exang": "category", "slope": "category", "thal": "category", "target": "category"})
```
<b>Handling missing data</b>

* For numeric data we decided to imputate the mean in order to keep the consistency of the data and not loose those registers.
* Fur categorical data, in order to avoid possible bias, we decided to drop the two registers that were NAs taking into account that putting an arbitrary value there could alter our analysis.

```py linenums="1" title="Handling missing data"
#Imputation and dropping
df["ca"] = df["ca"].fillna(df["ca"].mean()) #Filling with the mean 
df.dropna(axis=0,how="any",inplace=True) #Dropping qualitative missing data
```
## Descriptive statistics and variability across groups


