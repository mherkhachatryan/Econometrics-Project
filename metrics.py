from sklearn.metrics import average_precision_score, classification_report, log_loss
import sklearn
from scikitplot import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def show_metrics(y_test, y_predict, y_predict_proba):
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_predict)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_predict).ravel()

    recall = tp / float(tp + fn)

    print('Recall or Sensitivity : {0:0.4f}'.format(recall))

    precision = tp / float(tp + fp)

    print('Precision : {0:0.4f}'.format(precision))

    print("F-1 Score: ", sklearn.metrics.f1_score(y_test, y_predict))
    print("ROC-AUC: ", roc_auc)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_predict, pos_label=2)

    pr_auc = sklearn.metrics.auc(fpr, tpr)
    print("PR-AUC: ", pr_auc)
    metrics.plot_confusion_matrix(y_test, y_predict, normalize=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    metrics.plot_roc(y_test, y_predict_proba, ax=ax[0], classes_to_plot="1")
    metrics.plot_precision_recall_curve(y_test, y_predict_proba, ax=ax[1])
    f = plt.gcf()
    plt.show()
