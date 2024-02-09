# Project specification

## Proposal:
For applying our knowledge in machine learning models, we are interested in studying the application of
the data science field and state-of-the-art techniques within the health sciences frame. In this occasion, we want to approach the diagnosis of cardiovascular diseases, particularly, coronary artery disease (CAD) since they remain a leading cause of mortality worldwide. Thus, this project aims to leverage the "Heart Disease" database from the paper "International application of a new probability algorithm for the diagnosis of coronary artery disease" to explore and implement advanced classification algorithms for diagnosis.

## Database specification:
For our purpuse we used the available data in <a href="https://archive.ics.uci.edu/dataset/45/heart+disease" target="_blank" rel="noopener"> Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. </a>

### Variables and target:
The database uses a total of 13 explanatory variables and a single multicategory dependent variable. These exogenous attributes are aligned with the aim of the paper in the diagnosis of CAD and have been identified as most relevant in the identification on an early stage of the illness. Additionally, we used the Cleveland default set for aour aim.

<b> Explanatory variables: </b>
<ol>
<li>age: Age of the patient</li>
<li>sex: Sex of the patient (Male, Female)</li>
<li>cp: Chest pain type (Categorical with 4 levels-Type 1, Type 2, Type 3 and Type 4) Type 1:typical angina <li>Type 2:atypical angina Type 3:non-anginal pain Type 4:asymptomatic</li>
<li>trestbps: Resting blood pressure-in mm Hg on admission to the hospital(Continuous)</li>
<li>chol: Serum cholesterol in mg/dl (Continuous)</li>
<li>fbs: Fasting blood sugar > 120 mg/dl (True,False)</li>
<li>restecg: Resting electrocardiographic results (N(Normal), L1(Level 1), L2(Level 2))</li>
<li>thalach: Maximum heart rate achieved (Continuous)</li>
<li>exang: Exercise induced angina (Yes, No)</li>
<li>oldpeak: ST depression induced by exercise relative to rest (Continuous)</li>
<li>slope: The slope of the peak exercise ST segment (Up, Flat, Down)</li>
<li>ca: Number of major vessels (0-3) colored by flourosopy (0, 1, 2, 3)</li>
<li>thal: The heart status as retrieved from Thallium test (N(normal),FD(fixed defect), RD(reversible defect)</li>
</ol>

Plain codeblock:
```py linenums="1" title="Adding title" hl_lines="2 3"
import pandas as pd
# Some comment
print("Something")
```


## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
