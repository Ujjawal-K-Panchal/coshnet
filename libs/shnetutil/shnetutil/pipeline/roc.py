import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
#from sklearn.utils.fixes import signature  #scikit-learn .20.3
from funcsigs import signature	#scikit-learn .22
from scipy import interp

# Compute per-class ROC curve and ROC area for each class
def roc4classes(y_test, y_score, kPlot=True):
	shape = y_test.shape
	n_classes = 2 if len(shape) == 1 else shape[1]

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	return roc_auc
    
# Plot of a ROC curve for a specific class
def plotROC(y_test, y_score, kPlot=True, kSavePNG=''):
	fpr, tpr, thresholds = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")

	if kSavePNG != '':
		plt.savefig(kSavePNG)
	if kPlot:
		plt.show()

	return roc_auc

# Plot of a PR curve for a specific class
def plotPrecisionRecall(y_test, y_score, kPlot=True, kSavePNG=''):
	precision, recall, thresholds = precision_recall_curve(y_test, y_score)

	average_precision = average_precision_score(y_test, y_score)
	print('Average precision-recall score: {0:0.2f}'.format(average_precision))

	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
	               if 'step' in signature(plt.fill_between).parameters
	               else {})
	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
		
	if kSavePNG != '':
		plt.savefig(kSavePNG)
	if kPlot:
		plt.show()
