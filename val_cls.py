import numpy as np
from tqdm import tqdm

from utils import metrics
import torch

@torch.no_grad()
def validate(model, clsfier, val_loader, criterion):
    model.eval()
    clsfier.eval()

    device = next(model.parameters())[0].device
    gts = {i: [] for i in range(0, val_loader.dataset.n_classes)}
    preds = {i: [] for i in range(0, val_loader.dataset.n_classes)}
    losses = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = clsfier(outputs)

            losses.append(criterion(outputs, labels.float()).item())

            probs = torch.sigmoid(outputs)
            pred = probs.squeeze().data.cpu().numpy()
            gt = labels.float().squeeze().data.cpu().numpy()

            for label in range(0, val_loader.dataset.n_classes):
                gts[label].extend(gt[:, label])
                preds[label].extend(pred[:, label])

    auroc_dict, aupr_dict = metrics.auroc_aupr_scores(np.stack(list(gts.values()), axis=1),
                                                      np.stack(list(preds.values()), axis=1),
                                                      average_types=['macro', 'weighted'])

    model.train()
    clsfier.train()
    return aupr_dict, auroc_dict, np.mean(losses)