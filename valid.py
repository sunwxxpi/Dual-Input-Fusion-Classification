import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model import get_model_output


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)

    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += 1

    return conf_matrix


def calculate_metrics(cm, class_num):
    specificity = []
    sensitivity = []
    precision = []

    for i in range(class_num):
        true_negative = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        false_positive = cm[:, i].sum() - cm[i, i]
        false_negative = cm[i, :].sum() - cm[i, i]
        true_positive = cm[i, i]

        specificity.append(true_negative / (true_negative + false_positive + 1e-6))
        sensitivity.append(true_positive / (true_positive + false_negative + 1e-6))
        precision.append(true_positive / (true_positive + false_positive + 1e-6))

    avg_specificity = sum(specificity) / class_num
    avg_sensitivity = sum(sensitivity) / class_num
    avg_precision = sum(precision) / class_num
    avg_recall = avg_sensitivity
    f1score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)

    return f1score, avg_specificity, avg_sensitivity, avg_precision


def validate(config, model, val_loader, criterion):
    device = next(model.parameters()).device
    model.eval()

    print("START VALIDATION")

    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.class_num, config.class_num)).to(device)

    with tqdm(total=len(val_loader), desc="Validation", unit='Batch') as pbar:
        with torch.no_grad():
            for pack in val_loader:
                images = pack['imgs'].to(device)
                if images.shape[1] == 1:
                    images = images.expand((-1, 3, -1, -1))
                masks = pack['masks'].to(device)
                elastograms = pack['elastograms'].to(device)
                labels = pack['labels'].to(device)

                output = get_model_output(config, model, images, masks, elastograms)

                loss = criterion(output, labels)
                epoch_loss += loss.item() * images.size(0)  # accumulate loss over batch

                pred = output.argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_score.extend(output.softmax(dim=1).cpu().numpy())  # Apply softmax to get probabilities

                cm = confusion_matrix(pred.cpu(), labels.cpu(), cm)

                # Update progress bar
                pbar.update(1)

    avg_epoch_loss = epoch_loss / len(val_loader.dataset)

    # Calculate metrics
    acc = cm.diag().sum() / cm.sum()
    f1score, avg_specificity, avg_sensitivity, avg_precision = calculate_metrics(cm, config.class_num)

    # Compute AUC for each class and average
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if config.class_num > 2:
        # For multiclass, y_true should be in shape (n_samples,) and y_score in shape (n_samples, n_classes)
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    else:
        # For binary, y_true should be in shape (n_samples,) and y_score in shape (n_samples,)
        y_true = y_true.flatten()
        y_score = y_score[:, 1]  # Use the probability of the positive class
        auc = roc_auc_score(y_true, y_score)

    return [avg_epoch_loss, acc, f1score, auc, avg_sensitivity, avg_precision, avg_specificity]