import pandas as pd
import numpy as np
def decision_tree(x,y):
    import pandas as pd
    import numpy as np
    x=pd.DataFrame(x)
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion='gini',max_depth=2, min_samples_split=2, min_samples_leaf=2,
                                min_weight_fraction_leaf=0.05, max_features=None, max_leaf_nodes=None,
                                 min_impurity_split=None,random_state=100)
    ###########避免循环操作，简化运行时间###########
    model = clf.fit(x,y)
    return model
def plot_auc(y_predict,y_test):
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict)
    print(fpr, tpr, threshold)
    rocauc = metrics.auc(fpr, tpr)
    print(rocauc)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))  # 只能在这里面设置
    plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % rocauc)
    plt.legend(loc='lower right', fontsize=14)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # 增加坐标轴刻度#
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('TPR', fontsize=16)
    plt.xlabel('FPR', fontsize=16)
    plt.title('ROC CURVE', fontsize=12, color='red')
    plt.show()
def tree_deploy(tree):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.figure(figsize=(14, 12))
    plot_tree(tree, feature_names=x.columns, filled=True, rounded=True)
    plt.show()
def confussion_matrix(y_predict,y_test):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_test,y_predict)
def classification_report(y_predict,y_test):
    from sklearn.metrics import classification_report
    report=classification_report(y_test,y_predict,output_dict=True)
    return report
data=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')\
[:300]
x=data[['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']]
y=data['SeriousDlqin2yrs']
tree=decision_tree(x,y)
tree_deploy(tree)
x_predict=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')[1300:1340][['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']]
y_predict=tree.predict(x_predict)
y_true=pd.read_csv('C:/Users/任英睿/PycharmProjects/pythonProject/Financial_risk/TrainData.csv',encoding='gbk')[1300:1340]['SeriousDlqin2yrs']
plot_auc(y_predict,y_true)
data_confusion=confussion_matrix(y_predict,y_true)
print(data_confusion)
pd.DataFrame(data_confusion).to_excel('C:/Users/任英睿/PycharmProjects/pythonProject/cart树改进/data_confusion_2.xlsx',index=False)
report=classification_report(y_predict,y_true)
print(report)
pd.DataFrame(report).to_excel('C:/Users/任英睿/PycharmProjects/pythonProject/cart树改进/report_2.xlsx',index=False)
