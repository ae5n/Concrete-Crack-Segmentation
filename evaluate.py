from data import *
from configs import *
from loss import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(101)
tf.random.set_seed(101)

(train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_data(PATH)

print(f'Testing Data: {len(test_images)}')

test_dataset = test_data(test_images, test_masks)

def evaluate(MODEL):
	custom_objects={'dice_coef':dice_coef, 'bce_dice_loss':bce_dice_loss}
	model = load_model(MODEL, custom_objects=custom_objects)
	all_true_masks = []
	all_pred_masks = []
	for image, mask in test_dataset:
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
	p = precision_score(true_masks.ravel(), pred_masks.ravel().round(), pos_label=1)
	r = recall_score(true_masks.ravel(), pred_masks.ravel().round(), pos_label=1)
	f1_ = f1_score(true_masks.ravel(), pred_masks.ravel().round(), pos_label=1)
	p_ , r_ , _ = precision_recall_curve(true_masks.ravel(), pred_masks.ravel())
	acc = accuracy_score(true_masks.ravel(), pred_masks.ravel().round())
	c = confusion_matrix(true_masks.ravel(), pred_masks.ravel().round(), normalize='all')
	model_name = MODEL.split('/')[-1].split('.')[-2]
	print(f"Model Name: {model_name}\n")
	print(f'Precision = {p}\nRecall = {r}\nF1 Score = {f1_}\nAUC = {auc_}\nACC = {acc}')
	return model_name, p, r, f1_, fpr, tpr, auc_, p_, r_, acc, c


