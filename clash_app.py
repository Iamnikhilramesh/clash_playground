#Required libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import precision_score,recall_score,f1_score
from numpy.random import seed
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

#config
st.set_option('deprecation.showPyplotGlobalUse', False)

#Data prepration function
def data_prep(df1):
    try:
        if "Issue Description" in df1:
            df = []
            df = pd.DataFrame(df)
        
            #split the data value columns in issue description
            new = df1["Issue Description"].str.split(":", n = 1, expand = True) 
            df1["required"]= new[1]
    
            #Split the above required column 
            new = df1["required"].str.split(",", n = 5, expand = True) 
            df1["component1"]= new[0] 
            df1["component2"]= new[1]
            df1["x"]= new[2]
            df1["y"]= new[3]
            df1["z"]= new[4]
            df1["volume"]= new[5]
            
            #new dataframe with split values in component1 columns
            new = df1["component1"].str.split(".", n = 1, expand = True) 
            df1['comp1'] = new[0]
            df1['distance1'] = new[1]
            
            #new dataframe with split values in distance columns
            new = df1["distance1"].str.split(".", n = 2, expand = True) 
            df1['distance_1'] = new[0]
            df1['distance1_2'] = new[1]
            df1['distance'] = df1['distance_1'].str.cat(df1['distance1_2'], sep =".") 

            #new dataframe with split values in component2 columns
            new = df1["component2"].str.split(".", n = 1, expand = True) 
            df1['comp2'] = new[0]
            df1['distance2'] = new[1]
            
            #X_axis data extract withour the measure
            new = df1["x"].str.split("m", n = 1, expand = True) 
            df1['x_axis'] = new[0]
            
            #y_axis data extract withour the measure
            new = df1["y"].str.split("m", n = 1, expand = True) 
            df1['y_axis'] = new[0]
            
            #z_axis data extract withour the measure
            new = df1["z"].str.split("m", n = 1, expand = True) 
            df1['z_axis'] = new[0] 
            
            #Volume extract without the measure
            new = df1["volume"].str.split("m", n = 1, expand = True) 
            df1['v'] = new[0]
            
            #Extract the first clashed componenet
            new = df1["comp1"].str.split("\n", n = 1, expand = True) 
            df1['component1'] = new[1]
            
            #Extract the volume
            new = df1["v"].str.split(",", n = 1, expand = True) 
            df1['vol'] = new[1]
            
            #Keep only required column in the new dataframe as df and convert the column datatype to numeric
            if "x_axis" in df1:
                df["x_axis"] = pd.to_numeric(df1["x_axis"])
                if "y_axis" in df1:
                    df["y_axis"] = pd.to_numeric(df1["y_axis"])
                    if "z_axis" in df1:
                        df["z_axis"] = pd.to_numeric(df1["z_axis"])
                        if "vol" in df1:
                            df["vol"] = pd.to_numeric(df1["vol"])
                            if "distance2" in df1:
                                df["distance2"] = pd.to_numeric(df1["distance2"])
                                if "distance1" in df1:
                                    df["distance1"] = pd.to_numeric(df1["distance"])
            #feature engineering for ID column 
            df1["Dupli"] = df1.ID.duplicated(keep=False)
            #Encode the values that are not numbers
            #categorical column is encoded using cat.code function.
            df["Status_cat"] = df1.Status.astype('category').cat.codes
            df["Location_cat"] = df1.Location.astype('category').cat.codes
            df["component2_cat"] = df1.comp2.astype('category').cat.codes
            df["component1_cat"] = df1.component1.astype('category').cat.codes
            df["dup_cat"] = df1.Dupli.astype('category').cat.codes
            df["ID"] = df1["ID"]
            df["Discipline"] = df1["Discipline"]
            df['Phase'] = 1
            df = df.dropna(axis = 0, how ='any')
            return df
    except Exception as e:
        print(e)
#a function to display all the content in the side bar
def classifier(classifier_name):
    params =dict()
    if classifier_name == "Decision Tree Classifier":
        max_depth = st.sidebar.slider("Max Deepth",1,100)
        criterion = st.sidebar.selectbox("Select Criterion",("gini","entropy"))
        splitter = st.sidebar.selectbox("Select Splitter",("best","random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
    elif classifier_name == "Random Forest Classifier":
        n_estimators = st.sidebar.slider("N Estimators",1,100)
        criterion = st.sidebar.selectbox("Select Criterion",("gini","entropy"))
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
    elif classifier_name == "Gradient Boosting Classifier":
        n_estimators = st.sidebar.slider("N Estimators",1,100)
        criterion = st.sidebar.selectbox("Select Criterion",("friedman_mse","mse","mae"))
        loss = st.sidebar.selectbox("Select Loss",("deviance", "exponential"))
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["loss"] = loss
    elif classifier_name == "Logistic Regression":
        penalty = st.sidebar.selectbox("Select penalty",("l1","l2","elasticnet"))
        params["penalty"] = penalty
    return params

#a function to perform all the classifier tasks
def get_classifier(classifier_name,params):
    if classifier_name == "Decision Tree Classifier":
        clf = DecisionTreeClassifier()
    elif classifier_name == "Random Forest Classifier":
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],criterion=params["criterion"] ,random_state= 123)
    elif classifier_name == "Gradient Boosting Classifier":
        clf = GradientBoostingClassifier()
    elif classifier_name == "Logistic Regression":
        clf = LogisticRegression()
    return clf

#Application title and info
st.title("Model selection Playground")
st.write("Utilizing Machine Learning to Detect Non-pseudo/pseudo Design Conflicts in BIM Models")

#sidebar title
st.sidebar.header("Application Settings")
st.sidebar.subheader("Upload CSV or Excel file.")

#upload the file using sidebar and file uploader
uploaded_file = st.sidebar.file_uploader(label="",type=['csv','xlsx'])
class_names = ['Pseudo Clash', 'Non-Pseudo Clash']
#condition to upload a file and read the file
global df
if uploaded_file is not None:
    print(uploaded_file)
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file,engine='openpyxl')

#Display the content of the file
try:
    st.title("Original File contents") 
    st.write(df)
    #call the data prepration function
    df = data_prep(df)
    st.title("Normalized Data which is ready for the model") 
    st.write(df)
    #classifiers
    classifier_name = st.sidebar.selectbox("Select Classifier",("Random Forest Classifier", "Gradient Boosting Classifier","Decision Tree Classifier","Logistic Regression","Sequential"))
    #split x and y data
    x=df.drop(["Status_cat"],axis=1)
    y=df["Status_cat"]
    #split data, fit to classifier
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=123)
    if classifier_name != "Sequential":
        params = classifier(classifier_name)
        clf = get_classifier(classifier_name,params)
        
        clf.fit(x_train,y_train)
        y_predict = clf.predict(x_test)
        st.subheader("Confusion Matrix") 
        plot_confusion_matrix(clf, x_test, y_test, display_labels=class_names)
        st.pyplot()

        st.subheader("ROC Curve") 
        plot_roc_curve(clf, x_test, y_test)
        st.pyplot()

        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(clf, x_test, y_test)
        st.pyplot()
        

    elif classifier_name == "Sequential":
        # set random seed for reproducibility
        seed(42)

        model = Sequential()
        lyrs = [8]
        # create first hidden layer
        model.add(Dense(lyrs[0], input_dim=x_train.shape[1], activation='linear'))
        lyrs = [8]
        # create additional hidden layers
        for i in range(1,len(lyrs)):
            model.add(Dense(lyrs[i], activation='linear'))

        # add dropout, default is none
        model.add(Dropout(0.0))

        # create output layer
        model.add(Dense(1, activation='sigmoid'))  # output layer

        #Compile
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        # train model on full train set, with 80/20 CV split
        model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=0)

        
        y_predict = model.predict_classes(x_test)
        classifier_name = "Sequential"


    #evaluation metrics
    accuracy = accuracy_score(y_test,y_predict)
    precision = precision_score(y_test, y_predict, average='macro')
    recall = recall_score(y_test, y_predict, average='macro')
    f_score = f1_score(y_test, y_predict, average='macro')

    #Results
    st.title("Results")
    #display the classifier name and accuracy
    st.write(f"Classifier = {classifier_name}")
    st.write(f"Accuracy = {accuracy}")
    st.write(f"Precision = {precision}")
    st.write(f"Recall = {recall}")
    st.write(f"F1_score = {f_score}")



except Exception as e:
    print(e)
    st.write("Please upload a file to display the contents")





