import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

@torch.no_grad()
def get_probs_and_preds(model, df, device, transform, batch_size=32):
    from dataset import BottleDataset
    ds = BottleDataset(df, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for imgs, labels in dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]     # P(defective)
        preds = (probs > 0.5).long().cpu()
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_probs, all_preds, all_labels

def report(model, test_df, device, transform):
    probs, preds, labels = get_probs_and_preds(model, test_df, device, transform)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
    }