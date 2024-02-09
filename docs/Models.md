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
Before running our models we decided to calculate some descriptive statistics and variability test across groups to be sure that our findings were not biased by inconsistencies. For our descriptive statistics we found an approximate normal behaviour across variables. Additionally, we tested if there was a behavioural difference between men and women, and below and above the threshold of 65 (age in which an increase in cardiovascular attacks is expected).

Finally, we conclude this section by running a correlation matrix to understand if there was a risk of collinearity that we should handle before running our models.

## Running, testing, and comparing models.
To sum up, we create a function that allowed us to compare with the same sample and target variables the fit and performance of 5 models:
* Regression: Linear discriminant model, logistic regression, and probit regression.
* Classification trees: Random forest.
* Generalized linear classifiers: Supported Vector Machine.

This function also covered the topics of training and testing, cross validation, one-hot encoding, and scaling.

```py linenums="1" title="Creating and comparing models"
## Function

def ModelPerformance(data, response):
    """
    ModelPerformance() function gets the data frame and the response variable from the user 
    and aims to fit the best model out of Linear Discriminant Model (ldm_model), Logistic Regression Model (log_model),
    Probit Regression Model (probit_model), Random Forest Model (rf_model) and Support Vector Machines Model (svm_model).
    First the function creates training and testing data of explanatory variables as well as the response with
    train_test_split() function where the test_size is twenty percent of the entire data frame.
    Using Pipeline(), the function normalizes the numeric explanatory variables with StandardScaler() while creating 
    dummy variables with OneHotEncoder() for categorical variables. Then using kFold() technique data frame is splitted 
    into five folds, therefore n_splits is 5. Pipeline() function then creates the models with the preprocesses and 
    classifier as the each model.
    
    Later on, the training data is fit to all five models with .fit(). The function extracts  
    information that are Cross-Validation accuracy scores with cross_val_score() and have a prediction on test data with
    .predict(). From the test data the function extracts accuracy with accuracy_score() and using re precision, recall 
    and F1-scores of each model from the classification_report(). The function first presents the the Cross-Validation 
    accuracy scores cv_scores_(model name) and Cross-Validation standard deviations of all five models in two seperate 
    bar charts plt.bar() from pyplot. From the Cross-Validation results we can interpret how is the accuracy and fluctions
    for the each model on training data and it can help us analyze an overfit in the results. There is also a data frame 
    provided for the Cross-Validation results which is cv_scores. Using .predict() we can predict the outcome values for 
    test data which are Y_pred_(model name).The function also provides us a radar chart from plotly to analyze which model
    is best suiting for the the data frame with four metrics that are  accuracy, precision, recall and F1-scores. From the 
    radar chart using .add_trace() we can compare the four different aspects to pick the model for prediction. Finally, the
    function provides a data frame for these metrics of each model to visualize each result which is model_perf.
    """
    # Preprocessing
    X = data.loc[:, data.columns[data.columns != response]] 
    Y = data[response]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)

    numeric_features = data.columns[data.dtypes == "int64"]
    categorical_features = data.columns[(data.dtypes == "category") & (data.columns != response)]
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                               ('cat', categorical_transformer, categorical_features)])
    
    folds = KFold(n_splits=5, shuffle=True, random_state=13)
    
    # Linear Discriminant Model
    ldm_model = Pipeline(steps=[('preprocessor', preprocessor), 
                            ('classifier', LinearDiscriminantAnalysis())])
    ldm_model.fit(X_train, Y_train)
    cv_scores_ldm = cross_val_score(ldm_model, X_train, Y_train, cv=folds, scoring='accuracy')
    Y_pred_ldm = ldm_model.predict(X_test)
    accuracy_ldm = round(accuracy_score(Y_test, Y_pred_ldm),3)
    report_ldm = classification_report(Y_test, Y_pred_ldm)
    precision_ldm = float(re.search(r'weighted avg\s+([\d.]+)', report_ldm).group(1))
    recall_ldm = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_ldm).group(2))
    f1_score_ldm = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_ldm).group(3))
    
    # Logistic Regression Model
    log_model = Pipeline(steps=[('preprocessor', preprocessor), 
                            ('classifier', LogisticRegression())])

    log_model.fit(X_train, Y_train)
    cv_scores_log = cross_val_score(log_model, X_train, Y_train, cv=folds, scoring='accuracy')
    Y_pred_log = log_model.predict(X_test)
    accuracy_log = round(accuracy_score(Y_test, Y_pred_log),3)
    report_log = classification_report(Y_test, Y_pred_log)
    precision_log = float(re.search(r'weighted avg\s+([\d.]+)', report_log).group(1))
    recall_log = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_log).group(2))
    f1_score_log = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_log).group(3))
    
    # Probit Regression Model
    probit_model = Pipeline(steps=[('preprocessor', preprocessor), 
                                ('classifier',  LogisticRegression(solver='lbfgs', max_iter=1000, random_state=13))])
    probit_model.fit(X_train, Y_train)
    cv_scores_probit = cross_val_score(probit_model, X_train, Y_train, cv=folds, scoring='accuracy')
    Y_pred_probit = probit_model.predict(X_test)
    accuracy_probit = round(accuracy_score(Y_test, Y_pred_probit),3)
    report_probit = classification_report(Y_test, Y_pred_probit)
    precision_probit = float(re.search(r'weighted avg\s+([\d.]+)', report_probit).group(1))
    recall_probit = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_probit).group(2))
    f1_score_probit = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_probit).group(3))
    
    
    # Random Forest Model
    rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(random_state=13))])
    rf_model.fit(X_train, Y_train)
    cv_scores_rf = cross_val_score(rf_model, X_train, Y_train, cv=folds, scoring='accuracy')
    Y_pred_rf = rf_model.predict(X_test)  
    accuracy_rf = round(accuracy_score(Y_test, Y_pred_rf),3)
    report_rf = classification_report(Y_test, Y_pred_rf)
    precision_rf = float(re.search(r'weighted avg\s+([\d.]+)', report_rf).group(1))
    recall_rf = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_rf).group(2))
    f1_score_rf = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_rf).group(3))
    
    # Supported Vector Machine Model
    svm_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', SVC(random_state=13))])
    svm_model.fit(X_train, Y_train)
    cv_scores_svm = cross_val_score(svm_model, X_train, Y_train, cv=folds, scoring='accuracy')
    Y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = round(accuracy_score(Y_test, Y_pred_svm),3)
    report_svm = classification_report(Y_test, Y_pred_svm, zero_division=0)
    precision_svm = float(re.search(r'weighted avg\s+([\d.]+)', report_svm).group(1))
    recall_svm = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_svm).group(2))
    f1_score_svm = float(re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_svm).group(3))
    
    # Cross Validation mean score and standard deviation results
    
    cv_scores = pd.DataFrame({"CV Mean Score":[np.mean(cv_scores_ldm),np.mean(cv_scores_log),np.mean(cv_scores_probit),
                                               np.mean(cv_scores_rf),np.mean(cv_scores_svm)],
                             "CV Standard Deviation":[np.std(cv_scores_ldm),np.std(cv_scores_log),np.std(cv_scores_probit),
                                                 np.std(cv_scores_rf),np.std(cv_scores_svm)]})
    cv_scores.index = ["LDM","LOG","PRO","RF","SVM"]
    
    print("Cross Validation Results: \n")
    
    plt.bar(["LDM","LOG","PRO","RF","SVM"], [np.mean(cv_scores_ldm),np.mean(cv_scores_log),np.mean(cv_scores_probit),
            np.mean(cv_scores_rf),np.mean(cv_scores_svm)], color="forestgreen")
    plt.title("Averages of Cross Validation Scores (Accuracy)")
    plt.xlabel('Models')
    plt.ylabel('Avg. CV Scores')
    plt.show()
    
    plt.bar(["LDM","LOG","PRO","RF","SVM"], [np.std(cv_scores_ldm),np.std(cv_scores_log),np.std(cv_scores_probit),
            np.std(cv_scores_rf),np.std(cv_scores_svm)], color="forestgreen")
    plt.title("Standard Deviations of Cross Validation Scores (Accuracy)")
    plt.xlabel('Models')
    plt.ylabel('Standard Deviations')
    plt.show()
    
    print(cv_scores,"\n")
    
    # Performance Analysis: Accuracy, Precision, Recall, F1-Score
    categories = ["Accuracy","Precision","Recall","F1-Score"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
      r=[accuracy_ldm,precision_ldm,recall_ldm,f1_score_ldm],
      theta=categories,
      fill='toself',
      name='LDM'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[accuracy_log,precision_log,recall_log,f1_score_log],
      theta=categories,
      fill='toself',
      name='LOG'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[accuracy_probit,precision_probit,recall_probit,f1_score_probit],
      theta=categories,
      fill='toself',
      name='PRO'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[accuracy_rf,precision_rf,recall_rf,f1_score_rf],
      theta=categories,
      fill='toself',
      name='RF'
    ))
    fig.add_trace(go.Scatterpolar(
      r=[accuracy_svm,precision_svm,recall_svm,f1_score_svm],
      theta=categories,
      fill='toself',
      name='SVM'
    ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True
    )
    print("Model Performance Analysis: \n")
    fig.show()
    model_perf = pd.DataFrame({"Accuracy":[accuracy_ldm,accuracy_log,accuracy_probit,accuracy_rf,accuracy_svm],
                               "Precision":[precision_ldm,precision_log,precision_probit,precision_rf,precision_svm],
                               "Recall":[recall_ldm,recall_log,recall_probit,recall_rf,recall_svm],
                               "F1-Score":[f1_score_ldm,f1_score_log,f1_score_probit,f1_score_rf,f1_score_svm]})
    model_perf.index = ["LDM","LOG","PRO","RF","SVM"]
    print(model_perf) 

    return 

ModelPerformance(df, "target")  
print(ModelPerformance.__doc__)
```

## Result
As a result, we find that, for our exercise, the SVM model was the best one to perform in the test split. Not so ever, in the training set we notice a slightly better approach with the Random Forest, it depicted a better accuracy in average and a lower standard deviation.

Thus, because we want to study the impact of our model in predicting the possibility of having a cancer diagnosis, we determine that the best model in this case is the SVM. Because, even though it was not the best performer in the training sample, it perform similar to the other models in terms of the average accuracy across folds and depicted a low variance.

