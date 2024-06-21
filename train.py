import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import dataset
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from config import config
from model import create_model, get_model_output
from valid import validate


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)

    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)

    return conf_matrix


def save_results(model_save_path, filename, epoch, loss, val_acc, spe, sen, auc, pre, f1score, mode='a'):
    with open(os.path.join(model_save_path, filename), mode) as f:
        f.write(f'Result: (Epoch {epoch})\n')
        f.write('Loss: %f, Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1 Score: %f' % (loss, val_acc, spe, sen, auc, pre, f1score))


def train(config, train_loader, test_loader, fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = create_model(model_name=config.model_name, img_size=config.img_size, class_num=config.class_num, drop_rate=0.1, attn_drop_rate=0.1,
                         patch_size=config.patch_size, dim=config.dim, depth=config.depth, num_heads=config.num_heads,
                         num_inner_head=config.num_inner_head, mode=config.mode)
    model = model.to(device)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2).to(device) if config.loss_function == 'CE' else None

    # OPTIMIZER
    optimizer = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }[config.optimizer](model.parameters(), lr=config.lr)

    # SCHEDULER
    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    # TensorBoard WRITER
    writer = SummaryWriter(log_dir=f'./logs/{config.model_name}_{config.writer_comment}_{str(fold)}')

    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path, str(fold))
    
    best_val_acc = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        cm = torch.zeros((config.class_num, config.class_num))

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}", unit='Batch') as pbar:
            for pack in train_loader:
                images = pack['imgs'].to(device)
                if images.shape[1] == 1:
                    images = images.expand((-1, 3, -1, -1))
                masks = pack['masks'].to(device)
                labels = pack['labels'].to(device)

                output = get_model_output(config, model, images, masks)
                loss = criterion(output, labels)
                
                pred = output.argmax(dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                cm = confusion_matrix(pred.detach(), labels.detach(), cm)

                pbar.set_postfix(Loss=loss.item())
                pbar.update(1)

        lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_acc = cm.diag().sum() / cm.sum()
        print('Fold [%d], Epoch [%d/%d] - Avg Train Loss: %.4f' % (fold, epoch, config.epochs, avg_epoch_loss))

        # Log training metrics
        writer.add_scalar('Train/Avg Epoch Loss', avg_epoch_loss, global_step=epoch)
        writer.add_scalar('Train/Acc', train_acc, global_step=epoch)
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        if epoch % config.log_step == 0 or epoch == config.epochs:
            with torch.no_grad():
                result = validate(config, model, test_loader, criterion)

            # Log validation metrics
            val_loss, val_acc, f1score, auc, sen, pre, spe = result
            writer.add_scalar('Validation/Val Loss', val_loss, global_step=epoch)
            writer.add_scalar('Validation/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Validation/F1 Score', f1score, global_step=epoch)
            writer.add_scalar('Validation/AUC', auc, global_step=epoch)
            writer.add_scalar('Validation/Sen', sen, global_step=epoch)
            writer.add_scalar('Validation/Pre', pre, global_step=epoch)
            writer.add_scalar('Validation/Spe', spe, global_step=epoch)

            if epoch > (config.epochs // 5) and val_acc > best_val_acc:
                best_val_acc = val_acc
                print("=> saved best model")

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                if config.save_model:
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

                save_results(model_save_path, 'result_best.txt', epoch, val_loss, val_acc, f1score, auc, spe, sen, pre, 'w')

            if epoch == config.epochs:
                if config.save_model:
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))

                save_results(model_save_path, 'result_last_epoch.txt', epoch, val_loss, val_acc, f1score, auc, spe, sen, pre, 'a')

            writer.flush()
            
    writer.close()


if __name__ == '__main__':
    seed_torch(42)
    args = config()

    cv = KFold(n_splits=args.fold, random_state=42, shuffle=True)
    train_set = dataset.get_dataset(args.data_path, args.img_size, mode='train')
    test_set = dataset.get_dataset(args.data_path, args.img_size, mode='test')

    print(vars(args))
    args_path = os.path.join(args.model_path, args.model_name, args.writer_comment)

    if not os.path.exists(args_path):
        os.makedirs(args_path)
    with open(os.path.join(args_path, 'model_info.txt'), 'w') as f:
        f.write(str(vars(args)))

    print("START TRAINING")
    fold = 1
    for train_idx, test_idx in cv.split(train_set):
        print(f"\nCross Validation Fold {fold}")

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=6)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

        train(args, train_loader, test_loader, fold)

        fold += 1
        
# python train.py --data_path ./data/BUSI_2_class/train --class_num 2 --model_name hovertrans --writer_comment BUSI_2_class_img --mode img