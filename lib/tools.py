import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score


def train(model, criterion, optimizer, train_loader, device):
    print("Training...")
    model.train()
    losses = list()
    pred = list()
    gt = list()
    
    for data, label in tqdm(train_loader):
        data = data.float().to(device)
        label = label.long().to(device)
        output = model(data)
        loss = criterion(output, label)
        output_label = torch.max(output, 1).indices.cpu().detach().numpy()
        gt_label = label.cpu().detach().numpy()
        losses.append(loss.item())
        for i in range(output_label.shape[0]):
            pred.append(round(output_label[i]))
            gt.append(round(gt_label[i]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Loss:", np.mean(losses))
    print("Accuracy:", accuracy_score(gt, pred))
    print(classification_report(gt, pred))

def evaluate(model, criterion, val_loader, device):
    print("Evaluating...")
    model.eval()
    losses = list()
    pred = list()
    gt = list()
    
    for data, label in tqdm(val_loader):
        data = data.float().to(device)
        label = label.long().to(device)
        output = model(data)
        loss = criterion(output, label)
        output_label = torch.max(output, 1).indices.cpu().detach().numpy()
        gt_label = label.cpu().detach().numpy()
        losses.append(loss.item())
        for i in range(output_label.shape[0]):
            pred.append(round(output_label[i]))
            gt.append(round(gt_label[i]))
        
    print("Loss:", np.mean(losses))
    acc = accuracy_score(gt, pred)
    print("Accuracy:", acc)
    clf_report = classification_report(gt, pred)
    print(clf_report)
    
    return acc, clf_report
