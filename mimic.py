import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from xgboost import XGBClassifier, plot_importance
from sklearn import metrics
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap


data_path = 'data/mimic_preprocess.csv'


df = pd.read_csv(data_path)
# print(df.info())
# print(df.describe())
# print(df['die_in_hosp'].value_counts())

# feature, target 분리
X = df.drop('die_in_hosp', axis = 1)
y = df['die_in_hosp']

# train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=33)

# print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# baselines
xgb = XGBClassifier( eval_metric = 'logloss')
xgb.fit(X_train, y_train)

y_train_pred = xgb.predict(X_train)
y_train_prob = xgb.predict_proba(X_train)

y_val_pred = xgb.predict(X_val)
y_val_prob = xgb.predict_proba(X_val) 

print("\n[Baseline Model - Train]")
print(f"Accuracy: {metrics.accuracy_score(y_train, y_train_pred):.4f}")
print(f"AUC: {metrics.roc_auc_score(y_train, y_train_prob[:, 1]):.4f}")
print(metrics.classification_report(y_train, y_train_pred))

print("\n[Baseline Model - Validation]")
print(f"Accuracy: {metrics.accuracy_score(y_val, y_val_pred):.4f}")
print(f"AUC: {metrics.roc_auc_score(y_val, y_val_prob[:, 1]):.4f}")
print(metrics.classification_report(y_val, y_val_pred))


# class imbalance weight
neg = (y_train ==0).sum()
pos = (y_train ==1).sum()
scale_pos_weight = neg / pos

print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# hyperparameter tuning
def fitting_cv(model, param, kfold, train_input, train_target, scoring='roc_auc', n_jobs=1,tracking=True ):
    name, estimator = model
    
    if tracking:
        start_time = datetime.now()
        print(f"[{start_time}] Start parameter search for model '{name}'")
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=cv_idx, scoring=scoring, n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target )
        end_time = datetime.now()
        duration_time = (end_time - start_time).seconds
        print(f"[{end_time}] Finish parameter search for model '{name}' (time: {duration_time} seconds)\n")
        print()
    else:
        gridsearch = GridSearchCV(estimator=estimator, param_grid=param, cv=kfold, scoring=scoring, n_jobs=n_jobs)
        gridsearch.fit(train_input, train_target)
    
    return gridsearch   

model = ('XGB', XGBClassifier(
    eval_metric = 'logloss',
    random_state=33, 
    tree_method='hist', 
    scale_pos_weight=scale_pos_weight))
param = {
    'learning_rate': [0.05, 0.1, 0.3],
    'max_depth': [2, 3, 4],
    'n_estimators': [200, 300, 400],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}
cv_idx = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)

result = fitting_cv(model=model, param=param, kfold=cv_idx, train_input=X_train, train_target=y_train, n_jobs=1)

print(f"Best cross-validation score: {result.best_score_:.4f}")
print(f"Best parameters: {result.best_params_}")

tuned_xgb = result.best_estimator_

y_train_pred = tuned_xgb.predict(X_train)
y_train_prob = tuned_xgb.predict_proba(X_train)

y_val_pred  = tuned_xgb.predict(X_val)
y_val_prob = tuned_xgb.predict_proba(X_val)

print("\n[Tuned Model - Train]")
print(f"Accuracy: {metrics.accuracy_score(y_train, y_train_pred):.4f}")
print(f"AUC: {metrics.roc_auc_score(y_train, y_train_prob[:, 1]):.4f}")
print(metrics.classification_report(y_train, y_train_pred))

print("\n[Tuned Model - Validation]")
print(f"Accuracy: {metrics.accuracy_score(y_val, y_val_pred):.4f}")
print(f"AUC: {metrics.roc_auc_score(y_val, y_val_prob[:, 1]):.4f}")
print(metrics.classification_report(y_val, y_val_pred))

plt.figure(figsize=(10, 8))
plot_importance(tuned_xgb, max_num_features=15)
plt.title("Feature Importance (XGBoost)")
plt.savefig('output/feature_importance.png')
plt.show()

explainer = shap.Explainer(tuned_xgb, X_train)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val)
shap.summary_plot(shap_values, X_val, show=False)
plt.savefig('output/shap_summary.png')

# 최종 테스트 평가
y_test_pred = tuned_xgb.predict(X_test)
y_test_prob = tuned_xgb.predict_proba(X_test)

print("\n[Tuned Model - Test]")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_test_pred):.4f}")
print(f"AUC: {metrics.roc_auc_score(y_test, y_test_prob[:, 1]):.4f}")
print(metrics.classification_report(y_test, y_test_pred))

# confusion matrix plot for test
cm_test = metrics.confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test)')
plt.savefig('output/Confusion_Matrix(test).png')

