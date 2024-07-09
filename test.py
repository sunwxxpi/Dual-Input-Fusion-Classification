import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from model import create_model, get_model_output
from config import load_config


def plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}')
    
    plt.savefig(output_path)
    plt.close()


def load_model(model_path, config):
    model = create_model(model_name=config.model_name, img_size=config.img_size, class_num=config.class_num, drop_rate=0.1, attn_drop_rate=0.1,
                         patch_size=config.patch_size, dim=config.dim, depth=config.depth, num_heads=config.num_heads,
                         num_inner_head=config.num_inner_head, mode=config.mode)
    
    if torch.cuda.device_count() > 1:
        model.load_state_dict(torch.load(model_path))
        model = nn.DataParallel(model)
    else:
        model.load_state_dict(torch.load(model_path))
    
    return model


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = dataset.get_dataset(config.data_path, config.img_size, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=6)

    for fold in range(1, 6):
        all_labels = []
        all_preds = []
        model_path = os.path.join(config.model_path, config.model_name, config.writer_comment, str(fold), 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found at {model_path}")
            continue
        
        model = load_model(model_path, config)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for pack in tqdm(test_loader, desc=f'Testing Fold {fold}', unit='batch'):
                images = pack['imgs'].to(device)
                if images.shape[1] == 1:
                    images = images.expand((-1, 3, -1, -1))
                masks = pack['masks'].to(device)
                elastograms = pack['elastograms'].to(device)
                labels = pack['labels'].to(device)

                output = get_model_output(config, model, images, masks, elastograms)
                preds = output.argmax(dim=1).cpu().numpy()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(config.class_num)])
        conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')

        print(f'Fold {fold}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(class_report)

        output_path = os.path.join('./result', config.model_name, config.writer_comment, f'{fold}.png')
        plot_confusion_matrix(conf_matrix, [str(i) for i in range(config.class_num)], accuracy, f1, output_path)

if __name__ == '__main__':
    main()