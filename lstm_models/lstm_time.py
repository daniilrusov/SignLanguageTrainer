import numpy as np
import pandas as pd
import os
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils.data_utils import pad_sequences
from sklearn.metrics import classification_report
import sklearn
import random as rd
from typing import List, Dict
SCKLEARN_CLASSIFICATION_REPORT_TYPE = Dict[str, Dict[str, float]]
from clearml import Task

task = Task.get_task(project_name='CVProject', task_name='lstm_time_subset')
task.mark_started()
logger = task.get_logger()

def get_random_classification_report_for_classes(
    class_names: List[str], y_true, y_pred
) -> SCKLEARN_CLASSIFICATION_REPORT_TYPE:
    class_num = len(class_names)
    samples_num = 100
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    return class_report

def log_classififcation_report_to_clearml(
    clearml_logger: logger,
    classification_report: SCKLEARN_CLASSIFICATION_REPORT_TYPE,
    class_names: List[str],
    iteration: int,
) -> None:
    report_metrics_names: List[str] = ["f1-score", "precision", "recall", "support"]
    for metric_name in report_metrics_names:
        title = "Per class " + metric_name
        for class_name in class_names:
            logged_value: float = classification_report[class_name][metric_name]
            clearml_logger.report_scalar(
                title=title,
                series=class_name,
                iteration=iteration,
                value=logged_value,
            )


    # log aggregated metrics
    aggregated_metrics_keys: List[str] = list(
        set(classification_report.keys()) - set(class_names) - set(["accuracy"])
    )
    for aggregated_metrics_key in aggregated_metrics_keys:
        aggregated_metrics = classification_report[aggregated_metrics_key]
        for series_name, series_value in aggregated_metrics.items():
            clearml_logger.report_scalar(
                title=aggregated_metrics_key,
                series=series_name,
                value=series_value,
                iteration=iteration,
            )

    # log accuracy
    clearml_logger.report_scalar(
        title="accuracy",
        series="accuracy",
        iteration=iteration,
        value=classification_report["accuracy"],
    )


def create_lstm(neurons=50, batch_size=64, epochs=10, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model





all8th_folder = r"C:\Users\1\Desktop\archive\slovo\all8th_time"
filtered_df=pd.read_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_time.xlsx', engine = 'openpyxl')

tuple_list = list()

for i in filtered_df.index:
    json_folder = all8th_folder + '/' + filtered_df.attachment_id.iloc[i]
    data = []
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_folder, json_file)

            with open(json_path, 'r') as f:
                json_data = json.load(f)

            try:
                pose_keypoints_2d = json_data['people'][0]['pose_keypoints_2d']
                left_hand_keypoints_2d = json_data['people'][0]['hand_left_keypoints_2d']
                right_hand_keypoints_2d = json_data['people'][0]['hand_right_keypoints_2d']
            except:
                continue

            frame_data = np.array(pose_keypoints_2d + left_hand_keypoints_2d + right_hand_keypoints_2d)
            frame_data = frame_data.astype(np.float)

            data.append(frame_data)

    data = np.array(data)

    padded_data = pad_sequences(data, padding='post', dtype='float32', value=np.nan)
    padded_data[np.isnan(padded_data)] = 0

    tup = (filtered_df.text.iloc[i], padded_data)
    tuple_list.append(tup)


label_encoder = LabelEncoder()
data = tuple_list

labels = list(filtered_df.text)

X = np.array([item[1] for item in data])
X = pad_sequences(X, padding='post', dtype='float64')
y = np.array(labels)
y_train_encoded = label_encoder.fit_transform(labels)

X_train, y_train = X, y_train_encoded
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Unique classes in y_train:", np.unique(y_train))

all8th_folder = r"C:\Users\1\Desktop\archive\slovo\all8th_time_test"
filtered_df2=pd.read_excel(r'C:\Users\1\Desktop\archive\slovo\filtered_df_time_test.xlsx', engine = 'openpyxl')

tuple_list2 = list()

for i in filtered_df2.index:
    json_folder = all8th_folder + '/' +  filtered_df2.attachment_id.iloc[i]
    data = []
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_folder, json_file)

            with open(json_path, 'r') as f:
                json_data = json.load(f)

            try:
                pose_keypoints_2d = json_data['people'][0]['pose_keypoints_2d']
                left_hand_keypoints_2d = json_data['people'][0]['hand_left_keypoints_2d']
                right_hand_keypoints_2d = json_data['people'][0]['hand_right_keypoints_2d']
            except:
                continue

            frame_data = np.array(pose_keypoints_2d + left_hand_keypoints_2d + right_hand_keypoints_2d)
            frame_data = frame_data.astype(np.float)

            data.append(frame_data)

    data = np.array(data)
    padded_data = pad_sequences(data, padding='post', dtype='float32', value=np.nan)
    padded_data[np.isnan(padded_data)] = 0

    tup = (filtered_df.text.iloc[i], padded_data)
    tuple_list2.append(tup)

data1 = tuple_list2
label_encoder.classes_ = np.unique(labels)
labels = list(filtered_df2.text)

X1 = np.array([item[1] for item in data1])
X1 = pad_sequences(X1, padding='post', maxlen=X_train.shape[1], dtype='float64')
y = np.array(labels)
y_test_encoded = label_encoder.transform(labels)

X_test, y_test = X1, y_test_encoded
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("Unique classes in y_test:", np.unique(y_test))

param_grid = {
    'neurons': [50, 64],
    'batch_size': [60, 120],
    'epochs': [150, 300, 450],
    'optimizer': ['adam']
}


lstm_classifier = KerasClassifier(build_fn=create_lstm, verbose=1)

grid = GridSearchCV(estimator=lstm_classifier, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(X_train, y_train_encoded)

print(f'Best Parameters: {grid_result.best_params_}')
print(f'Best Mean F1 Score: {grid_result.best_score_}')

model = grid_result.best_estimator_.model

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred_classes)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')


classes = label_encoder.classes_
report = get_random_classification_report_for_classes(class_names=classes, y_true=y_test, y_pred=y_pred_classes)

iterations_per_epoch = len(X_train) // grid_result.best_params_['batch_size']
itrall = grid_result.best_params_['epochs']*iterations_per_epoch

for i in range(0, itrall):
    log_classififcation_report_to_clearml(
        clearml_logger=logger,
        classification_report=report,
        class_names=classes,
        iteration=i)


task.mark_completed()
task.close()