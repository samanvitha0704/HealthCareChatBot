import streamlit as st
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        st.write("You should take the consultation from a doctor.")
    else:
        st.write("It might not be that bad, but you should take precautions.")


def getDescription():
    global description_list
    with open(r"C:\Users\ambik\OneDrive\Documents\minor proj\MasterData\symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open(r"C:\Users\ambik\OneDrive\Documents\minor proj\MasterData\Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open(r"C:\Users\ambik\OneDrive\Documents\minor proj\MasterData\symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    st.write("-----------------------------------HealthCare ChatBot-----------------------------------")
    name = st.text_input("Your Name?")
    st.write("Hello, " + name)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv(r"C:\Users\ambik\OneDrive\Documents\minor proj\Data\Training.csv")
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        disease_input = st.text_input("Enter the symptom you are experiencing")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            if len(cnf_dis) > 0:
                conf_inp = st.selectbox("Select the symptom you meant", cnf_dis)
                disease_input = conf_inp
                break
        else:
            st.write("Enter a valid symptom.")
    while True:
        try:
            num_days = int(st.text_input("From how many days have you been experiencing it?", key="hello"))
            break
        except ValueError:
            st.write("Enter a valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            st.write("Are you experiencing any of the following symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = st.selectbox(syms, ["Yes", "No"])
                if inp == "Yes":
                    symptoms_exp.append(syms)
                elif inp == "No":
                    symptoms_exp.append("No " + syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                st.write("You may have " + present_disease[0])
                st.write(description_list[present_disease[0]])
            else:
                st.write("You may have " + present_disease[0] + " or " + second_prediction[0])
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            st.write("Take following measures:")
            for i, j in enumerate(precution_list):
                st.write(i + 1, ")", j)

    recurse(0, 1)


def main(severityDictionary, description_list, precautionDictionary, clf, cols, le, training, reduced_data):
    st.sidebar.title("Options")
    menu = st.sidebar.selectbox("Select an option", ["Home", "ChatBot"])

    if menu == "Home":
        st.title("HELPI")
        getInfo()
    elif menu == "ChatBot":
        st.title("HealthCare ChatBot")
        st.write("-----------------------------------HealthCare ChatBot-----------------------------------")
        st.write("Enter your symptoms and get possible diagnoses.")
        st.write("Please make sure to enter symptoms correctly and answer the questions accurately.")
        st.write("Let's get started!")

        tree_to_code(clf, cols)

    st.write("----------------------------------------------------------------------------------------")


if __name__ == "__main__":
    severityDictionary = dict()
    description_list = dict()
    precautionDictionary = dict()
    le = preprocessing.LabelEncoder()
    training = pd.read_csv(r"C:\Users\ambik\OneDrive\Documents\minor proj\Data\Training.csv")
    reduced_data = training.groupby(training['prognosis']).max()
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    le.fit(y)
    y = le.transform(y)
    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x, y)
    getDescription()
    getSeverityDict()
    getprecautionDict()
    main(severityDictionary, description_list, precautionDictionary, clf, cols, le, training, reduced_data)