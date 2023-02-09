# Import some important libraries going to use in this exercise
import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# Set title
st.title("My First Exercise")

# Adding image
image = Image.open("https://github.com/ansaridatascientist/first_exercise_st/blob/main/photo.png")
st.image(image, use_column_width = True)

# Set subtitle
st.write("""
        ## A Simple Data App with Streamlit
        """)
st.write("""
        ### Let's explore the different classifiers and datasets
        """)


dataset_name = st.sidebar.selectbox("Select Dataset", ("Breast Cancer", "Iris", "Wine"))
classifier_name = st.sidebar.selectbox("Select the Classifier Algorithm", ("SVM", "KNN"))

# Create a function which extract the dataset based on the selection
def get_dataset(name):
    data = None
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target
    
    return x, y
    
x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write("Shape of your dataset is: ", x.shape)
st.write("Unique target variables: ", len(np.unique(y)))

fig = plt.figure(figsize=(10,6))
plt.boxplot(x)
st.pyplot(fig)

fig = plt.figure(figsize=(10,4))
plt.hist(x)
st.pyplot(fig)

# Buiding our algorithm
def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == "SVM":
        C = st.sidebar.slider("C", 0.01, 15.0)
        params["C"] = C
    else:
        name_of_clf = "KNN"
        k = st.sidebar.slider("k", 1, 15)
        params["k"] = k
    return params
    
params = add_parameter(classifier_name)

# Accessing our classifier
def get_classifier(name_of_clf, params):
    clf=None
    if name_of_clf == "SVM":
        clf=SVC(C = params["C"])
    elif name_of_clf == "KNN":
        clf=KNeighborsClassifier(n_neighbors = params["k"])
    else:
        st.warning("you didn't selet any option, please select atleast one algorithm")
    return clf

clf = get_classifier(classifier_name, params)
 
# Spliting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.20,
                                                    random_state = 42)
clf.fit(x_train, y_train)

y_preds = clf.predict(x_test)
st.write(y_preds)
accuracy = accuracy_score(y_test, y_preds)
st.write("Classifier_Name: ", classifier_name)
st.write("Accuracy for your model is: ", accuracy)
    

    
