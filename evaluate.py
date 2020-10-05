from data import *
from configs import *
from loss import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def evaluate(MODEL):
	custom_objects={'dice_coef':dice_coef, 'bce_dice_loss':bce_dice_loss}
	model = load_model(MODEL, custom_objects=custom_objects)
	all_true_masks = []
	all_pred_masks = []
	for image, mask in test_data:
		pred_mask = model.predict(image)
		pred_mask = np.squeeze(pred_mask, -1)
		true_mask = np.squeeze(mask, -1)
		all_pred_masks.append(pred_mask)
		all_true_masks.append(true_mask)
	flat_true_masks = [item for sublist in all_true_masks for item in sublist]
	flat_pred_masks = [item for sublist in all_pred_masks for item in sublist]
	true_masks = np.array(flat_true_masks)
	pred_masks = np.array(flat_pred_masks)

	fpr, tpr, thresholds = roc_curve(true_masks.ravel(), pred_masks.ravel(), pos_label=1)
	auc_ = auc(fpr, tpr)
	p_ , r_ , _ = precision_recall_curve(true_masks.ravel(), pred_masks.ravel())

	report = classification_report(true_masks.ravel(), pred_masks.ravel().round(), output_dict=True)
	p = report['1.0']['precision']
	r = report['1.0']['recall']
	f1 = report['1.0']['f1-score']

	cm = confusion_matrix(true_masks.ravel(), pred_masks.ravel().round())
	model_name = MODEL.split('/')[-1].split('.')[-2]
	print(f"Model Name: {model_name}\nF1 Score = {f1}\nPrecision = {p}\nRecall = {r}\nAUC = {auc_}\nConfusion Matrix:\n{cm}")

	return model_name, fpr, tpr, auc_, cm

def ROC_Zoom(fpr1, tpr1, auc1, name1, fpr2, tpr2, auc2, name2):
	plt.figure(figsize=(8,6))
	plt.xlim(-.002, 0.1)
	plt.ylim(0.8, 1.005)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.grid(b=True, linestyle=':')
	plt.plot(fpr1, tpr1, label=f'{name1} (AUC = {round(auc1,3)})')
	plt.plot(fpr2, tpr2, label=f'{name2} (AUC = {round(auc2, 3)})')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve (zoomed in at the top left)')
	plt.legend(loc='best')
	plt.savefig('ROC_Zoom', dpi=300)
	plt.show()

def confusion_display(cm):
	labels = ['Background', 'Crack']
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	disp.plot(include_values=True, values_format='d', cmap='plasma', ax=None, xticks_rotation='horizontal')
	plt.xlabel('Predicted')
	plt.ylabel('Ground-truth')



