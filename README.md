# Coursework 2 - Machine learning Approaches to Space Mission Success

## Overview
This coursework explores the prediction of Space Mission success through the use of machine learning techniques. The same dataset is used throughout to ensure a fair comparison between approaches.

The coursework is structured into three parts:
- **Q1:** Traditional machine learning approach
- **Q2:** Neural network approach and comparison with Q1
- **Q3:** Investigation into how data choices affect neural network performance

The main focus of the coursework is on model choice, data handling, evaluation metrics, and interpretation of results.

## Dataset
The dataset used for this coursework is Space Mission Launches, which contains historical information on space missions such as:
- Organisation
- Location
- Rocket status
- Launch date
- Mission price
- Mission outcome

The task is a binary classification problem:
- 1 = Successful mission
- 0 = Unsuccessful mission

The dataset is located at:

**data/mission_launches_csv**

https://www.kaggle.com/datasets/sefercanapaydn/mission-launches

---
## Repository 
```
coursework-2/
├── data/
│   └── mission_launches.csv
├── py/
│   ├── __init__.py
│   └── functions.py
├── Q1_folder/
│   └── Q1.ipynb
├── Q2_folder/
│   └── Q2.ipynb
├── Q3_folder/
│   └── Q3.ipynb
├── dependencies.txt
└── README.md
```
shared data handling and preprocessing are implemented in **py/functions.py** and reused across all questions.

## Dependencies
All required python packages are listed in dependencies.txt

## Question Summary
### Q1 - Traditional Machine Learning
For predicting mission success, a logistic regression model is trained as a traditional baseline. The performance is evaluated using accuracy, ROC-AUC, and a confusion matrix.

### Q2 - Neural Network
A feedforward neural network is trained using the same dataset and preprocessing pipeline as Q1 to ensure a fair comparison. The results of this are compared with logistic regression in terms of performance and the complexity of the model.

### Q3 - Research Question
How do choices about data affect the performance of a neural network?

using the same dataset and preprocessing as previous questions, two experiments vary:
- the amount of training data
- class balance in the training set

performance is measured primarily using ROC-AUC.

---
## How to Run
**1.** Install dependencies

**2.** Launch Jupyter

**3.** Run notebooks:
- Q1_folder/Q1.ipynb
- Q2_folder/Q2.ipynb
- Q3_folder/Q3.ipynb

Each notebook runs top to bottom without any changes.


## Author
**[Lewis - UP2120028]**



