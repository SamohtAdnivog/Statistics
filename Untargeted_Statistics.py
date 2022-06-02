"""
Statistics
"""
# Importing Packages
from Statmethods import *
from Visualizations import *
import pandas as pd
from varname import nameof
import os
import time

st = time.time()

print(f"Packages are loaded..")

# Reading Datafiles
df = pd.read_csv("data_normalized.csv", index_col=0)
demo = pd.read_csv("Schizo_Studiendaten.csv", index_col=0)
deleting = 'Yes'  # Yes or No
method = "_Annotated"
age_border = 60

print(f"Shape of the inputted feature table: \t{df.shape}")
if deleting == 'Yes':
    print('Features between RT 0 and 1 min get dropped')
    df2 = df[df.index.str.startswith('0')]
    red_features = df2.index
    df.drop(red_features, inplace=True)
print(f"Shape of the cleaned feature table: \t{df.shape}")

# Filtering for Groups
# Single variable (6 Groups / 3 comparisons)
temporal = df.filter(like='T', axis=1)
frontal = df.filter(like='F', axis=1)
white = df.filter(like='W', axis=1)
cortex = df.filter(like='C', axis=1)
schizophrenia = df.filter(regex="^2", axis=1)
control = df.filter(regex="^1", axis=1)
# Two variables (12 Groups / 6 comparisons)
temporal_white = df.filter(like="TW", axis=1)
temporal_cortex = df.filter(like="TC", axis=1)
frontal_white = df.filter(like="FW", axis=1)
frontal_cortex = df.filter(like="FC", axis=1)
schizo_cortex = schizophrenia.filter(like='C', axis=1)
control_cortex = control.filter(like='C', axis=1)
schizo_white = schizophrenia.filter(like='W', axis=1)
control_white = control.filter(like='W', axis=1)
schizo_frontal = schizophrenia.filter(like='F', axis=1)
control_frontal = control.filter(like='F', axis=1)
schizo_temporal = schizophrenia.filter(like='T', axis=1)
control_temporal = control.filter(like='T', axis=1)
# Three variables (8 Groups / 4 comparisons)
frontal_white_schizo = frontal_white.filter(regex="^2", axis=1)
frontal_white_control = frontal_white.filter(regex="^1", axis=1)
frontal_cortex_schizo = schizo_cortex.filter(like='F', axis=1)
frontal_cortex_control = control_cortex.filter(like='F', axis=1)
temporal_white_schizo = schizo_white.filter(like='T', axis=1)
temporal_white_control = control_white.filter(like='T', axis=1)
temporal_cortex_schizo = schizo_cortex.filter(like='T', axis=1)
temporal_cortex_control = control_cortex.filter(like='T', axis=1)

# Selecting male and female of controlgroup:
male_ind = []
female_ind = []
younger = []
older = []
for row, index in control.T.iterrows():
    for item, sex, age in zip(demo.index, demo['Sex'], demo['Age']):
        test = int(row[0:3])
        if test == item and sex == 0:
            female_ind.append(row)
        elif test == item and sex == 1:
            male_ind.append(row)
        if test == item and age < age_border + 1:
            younger.append(row)
        elif test == item and age > age_border:
            older.append(row)
younger = df.filter(younger)
older = df.filter(older)
female = df.filter(female_ind)
male = df.filter(male_ind)
# Selecting male and female of white_control:
younger_white = []
older_white = []
female_white = []
male_white = []
male_ind = []
female_ind = []
for row, index in control_white.T.iterrows():
    for item, sex, age in zip(demo.index, demo['Sex'], demo['Age']):
        test = int(row[0:3])
        if test == item and sex == 0:
            female_ind.append(row)
        elif test == item and sex == 1:
            male_ind.append(row)
        if test == item and age < age_border + 1:
            younger_white.append(row)
        elif test == item and age > age_border:
            older_white.append(row)
younger_white = df.filter(younger_white)
older_white = df.filter(older_white)
female_white = df.filter(female_ind)
male_white = df.filter(male_ind)
# Selecting male and female of grey_control:
male_ind = []
female_ind = []
younger_cortex = []
older_cortex = []
for row, index in control_cortex.T.iterrows():
    for item, sex, age in zip(demo.index, demo['Sex'], demo['Age']):
        test = int(row[0:3])
        if test == item and sex == 0:
            female_ind.append(row)
        elif test == item and sex == 1:
            male_ind.append(row)
        if test == item and age < age_border + 1:
            younger_cortex.append(row)
        elif test == item and age > age_border:
            older_cortex.append(row)
younger_grey = df.filter(younger)
older_grey = df.filter(older)
female_grey = df.filter(female_ind)
male_grey = df.filter(male_ind)

# Selecting Groups:
pairs = [[frontal, temporal], [white, cortex], [schizophrenia, control],
         [temporal_white, temporal_cortex], [frontal_white, frontal_cortex],
         [temporal_white, frontal_white], [temporal_cortex, frontal_cortex],
         [schizo_cortex, control_cortex], [schizo_white, control_white],
         [schizo_frontal, control_frontal], [schizo_frontal, schizo_temporal],
         [schizo_temporal, control_temporal], [control_frontal, control_temporal],
         [frontal_white_schizo, frontal_white_control], [frontal_cortex_schizo, frontal_cortex_control],
         [temporal_white_schizo, temporal_white_control], [temporal_cortex_schizo, temporal_cortex_control],
         [frontal_white_schizo, temporal_white_schizo], [frontal_white_control, temporal_white_control],
         [frontal_cortex_schizo, temporal_cortex_schizo], [frontal_cortex_control, temporal_cortex_control],
         [control_white, control_cortex],
         [female, male], [female_white, male_white], [female_grey, male_grey],
         [younger, older], [younger_white, older_white], [younger_grey, older_grey]]
pairs_names = [[nameof(temporal), nameof(frontal)], [nameof(white), nameof(cortex)],
               [nameof(schizophrenia), nameof(control)],
               [nameof(temporal_white), nameof(temporal_cortex)], [nameof(frontal_white), nameof(frontal_cortex)],
               [nameof(temporal_white), nameof(frontal_white)], [nameof(temporal_cortex), nameof(frontal_cortex)],
               [nameof(schizo_cortex), nameof(control_cortex)], [nameof(schizo_white), nameof(control_white)],
               [nameof(schizo_frontal), nameof(control_frontal)], [nameof(schizo_frontal), nameof(schizo_temporal)],
               [nameof(schizo_temporal), nameof(control_temporal)], [nameof(control_frontal), nameof(control_temporal)],
               [nameof(frontal_white_schizo), nameof(frontal_white_control)],
               [nameof(frontal_cortex_schizo), nameof(frontal_cortex_control)],
               [nameof(temporal_white_schizo), nameof(temporal_white_control)],
               [nameof(temporal_cortex_schizo), nameof(temporal_cortex_control)],
               [nameof(frontal_white_schizo), nameof(temporal_white_schizo)],
               [nameof(frontal_white_control), nameof(temporal_white_control)],
               [nameof(frontal_cortex_schizo), nameof(temporal_cortex_schizo)],
               [nameof(frontal_cortex_control), nameof(temporal_cortex_control)],
               [nameof(control_white), nameof(control_cortex)],
               [nameof(female), nameof(male)], [nameof(female_white), nameof(male_white)],
               [nameof(female_grey), nameof(male_grey)],
               [nameof(younger), nameof(older)], [nameof(younger_white), nameof(older_white)], [nameof(younger_grey), nameof(older_grey)]]

# Preparing Frames:
comparison = pd.DataFrame(
    columns=["Comparison / CV Scores", 'Bayes Naive Classifier', 'k-nearest Neighbour', 'Random Forest',
             'Support Vector Machine'])

print(f"Groups are prepared..")
count = 1
# Making a lot of dataframes:
for pair, pair_names in zip(pairs, pairs_names):
    A = pair[0]
    B = pair[1]
    A_Name = pair_names[0]
    B_Name = pair_names[1]
    print(
        f"({np.round((time.time() - st), 2)} seconds) \t ({count}/{len(pairs)}) \t Comparing {A_Name} with {B_Name}...")
    count = count + 1
    classes = [A_Name, B_Name]
    X, y = settingVars(A, B)

    nombre = A_Name + ' vs. ' + B_Name
    path = nombre
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, nombre)

    ## Doing Statistics
    stats = stats_framing(A, B, df)
    stats.sort_values('p_value', inplace=True)
    stats.to_csv(filename + '.csv')
    df_complex = X.T
    df_complex['p_value'] = stats['p_value']
    df_complex.to_csv(filename + '_complex.csv')

    ## Doing Visualizations
    heatmap(X, y, stats, A_Name, B_Name, filename, 50)  # Important Features due to T-Tests
    PCA_Anal(X, y, A_Name, B_Name, classes, filename)

    ## ML Algorithmns
    gnb_cv_score = BNC(X, y)  # Bayes Naive Classifier
    knn_cv_score = KNN(X, y)  # k-nearest Neighbour
    rf_cv_score = RFC(X, y)  # Random Forest Classification
    svc_cv_score = lSVC(X, y)  # Support Vector Machine
    # Results saving in comparison
    loop_name = A_Name + ' vs. ' + B_Name + method
    comp = [loop_name, gnb_cv_score, knn_cv_score, rf_cv_score, svc_cv_score]
    complen = len(comparison.index)
    comparison.loc[complen] = comp
comparison.to_csv('machine_learning_results.csv')
heatmap_ml(method)
