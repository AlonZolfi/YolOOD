import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from tqdm import tqdm
import sklearn.covariance
from utils.load_utils import get_loaders


to_np = lambda x: x.data.cpu().numpy()


def get_odin_scores(loader, model, clsfier, method, T, noise):
    # get logits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bceloss = nn.BCEWithLogitsLoss(reduction="none")
    scores = []
    for images, _ in tqdm(loader):
        images = images.to(device)
        images.requires_grad_(True)

        nnOutputs = model(images)
        nnOutputs = clsfier(nnOutputs)

        # using temperature scaling
        preds = torch.sigmoid(nnOutputs / T)

        labels = torch.ones_like(preds) * (preds >= 0.5)
        labels = labels.float()

        # input pre-processing
        loss = bceloss(nnOutputs, labels)

        if method == 'max':
            idx = torch.max(preds, dim=1)[1].unsqueeze(-1)
            loss = torch.mean(torch.gather(loss, 1, idx))
        elif method == 'sum':
            loss = torch.mean(torch.sum(loss, dim=1))

        loss.backward()
        # calculating the perturbation
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        adv_inputs = torch.add(images.data, gradient, alpha=-noise)

        with torch.no_grad():
            nnOutputs = model(adv_inputs)
            nnOutputs = clsfier(nnOutputs)

            # compute odin score
            outputs = torch.sigmoid(nnOutputs / T)

            if method == "max":
                score = np.max(to_np(outputs), axis=1)
            elif method == "sum":
                score = np.sum(to_np(outputs), axis=1)

            scores.append(score)
    scores = np.hstack(scores)
    return scores


@torch.no_grad()
def sample_estimator(model, train_loader, num_exits, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = train_loader.dataset.n_classes
    list_features = [[[] for _ in range(num_classes)] for i in range(num_exits)]
    for i, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        target = target.to(device)

        out_features = model_feature_list(model, data, args.arch)
        for layer_num, layer_output in enumerate(out_features[-num_exits:]):
            layer_output = layer_output.view(layer_output.size(0), layer_output.size(1), -1)
            layer_output = torch.mean(layer_output, 2)

            # construct the sample matrix
            # use the training set labels(multiple) or set with the one with max prob
            indices_tuple = target.nonzero(as_tuple=True)
            for batch, label in zip(indices_tuple[0], indices_tuple[1]):
                list_features[-layer_num-1][label.item()].append(layer_output[batch.item()].view(1, -1))

    sample_class_mean = [[] for i in range(num_exits)]
    for i in range(num_exits):
        for j in range(num_classes):
            list_features[i][j] = torch.cat(list_features[i][j])
            sample_class_mean[i].append(torch.mean(list_features[i][j], 0))
        sample_class_mean[i] = torch.stack(sample_class_mean[i])

    precision = []
    for i in range(num_exits):
        X = []
        for j in range(num_classes):
            X.append(list_features[i][j] - sample_class_mean[i][j])
        X = torch.cat(X, 0)

        # find inverse
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        precision.append(temp_precision)

    return sample_class_mean, precision


def get_mahalanobis_score(model, clsfier, loader, pack, noise, method, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = args.n_classes
    sample_mean, precision = pack
    num_exits = len(sample_mean)
    model.eval()
    clsfier.eval()

    Mahalanobis = []
    for data, target in tqdm(loader):
        data = data.to(device)
        if noise > 0:
            data.requires_grad_(True)

        maha_score = []
        for exit_num in range(num_exits):
            layer_output = model_intermediate_forward(model, data, args.arch, -exit_num - 1)
            layer_output = layer_output.view(layer_output.size(0), layer_output.size(1), -1)
            layer_output = torch.mean(layer_output, 2)

            # compute Mahalanobis score
            gaussian_score = []
            for j in range(num_classes):
                batch_sample_mean = sample_mean[exit_num][j]
                zero_f = layer_output - batch_sample_mean
                term_gau = -torch.mm(torch.mm(zero_f, precision[exit_num]), zero_f.t()).diag()
                gaussian_score.append(term_gau.view(-1, 1))

            gaussian_score = torch.cat(gaussian_score, 1)

            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[exit_num].index_select(0, sample_pred)
            zero_f = layer_output - batch_sample_mean
            pure_gau = -torch.mm(torch.mm(zero_f, precision[exit_num]), zero_f.t()).diag()

            if noise > 0:
                # Input_processing
                loss = torch.mean(-pure_gau)
                loss.backward()

                with torch.no_grad():
                    gradient = torch.ge(data.grad, 0)
                    gradient = (gradient.float() - 0.5) * 2
                    tempInputs = torch.add(data, gradient, alpha=-noise)
                    noise_out_features = model_intermediate_forward(model, tempInputs, args.arch, -exit_num - 1)
                    noise_layer_output = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
                    noise_layer_output = torch.mean(noise_layer_output, 2)

                    # compute Mahalanobis score
                    gaussian_score = []
                    for j in range(num_classes):
                        batch_sample_mean = sample_mean[exit_num][j]
                        zero_f = noise_layer_output - batch_sample_mean
                        term_gau = -torch.mm(torch.mm(zero_f, precision[exit_num]), zero_f.t()).diag()
                        gaussian_score.append(term_gau.view(-1, 1))

                    gaussian_score = torch.cat(gaussian_score, 1)
                    maha_score.append(gaussian_score.max(dim=1)[0])
                data.grad.zero_()
                model.model.zero_grad()
            else:
                maha_score.append(pure_gau)
        maha_score = torch.stack(maha_score, 1).sum(dim=1)
        Mahalanobis.extend(to_np(maha_score))

    return Mahalanobis


def model_feature_list(model, x, arch):
    out_list = []
    if arch == "yolo_backbone":
        out = model.model[:3](x)
        out_list.append(out)
        out = model.model[3:5](out)
        out_list.append(out)
        out = model.model[5:7](out)
        out_list.append(out)
        out = model.model[7:9](out)
        out_list.append(out)
        out = model.model[9:](out)
        out_list.append(out)
    return out_list


def model_intermediate_forward(model, x, arch, exit_num):
    if arch == "yolo_backbone":
        out = model.model[:3](x)
        if exit_num == 0 or exit_num == -5:
            return out
        out = model.model[3:5](out)
        if exit_num == 1 or exit_num == -4:
            return out
        out = model.model[5:7](out)
        if exit_num == 2 or exit_num == -3:
            return out
        out = model.model[7:9](out)
        if exit_num == 3 or exit_num == -2:
            return out
        out = model.model[9:](out) # exit_num == 4 or exit_num == -1
        return out


@torch.no_grad()
def get_logits(loader, model, clsfier, args, k=20, name=None):
    logits_np = np.empty([0, loader.dataset.n_classes])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, (images, labels) in enumerate(tqdm(loader)):

        images = images.to(device)
        nnOutputs = model(images)
        nnOutputs = clsfier(nnOutputs)
        nnOutputs_np = to_np(nnOutputs.squeeze())

        logits_np = np.vstack((logits_np, nnOutputs_np))

    ## Compute the Score
    logits = torch.from_numpy(logits_np).to(device)
    outputs = torch.sigmoid(logits)
    if args.ood == "logit":
        if args.method == "max": scores = np.max(logits_np, axis=1)
        if args.method == "sum": scores = np.sum(logits_np, axis=1)
    elif args.ood == "energy":
        E_f = torch.log(1+torch.exp(logits))
        if args.method == "max": scores = to_np(torch.max(E_f, dim=1)[0])
        if args.method == "sum": scores = to_np(torch.sum(E_f, dim=1))
        if args.method == "topk":
            scores = to_np(torch.sum(torch.topk(E_f, k=k, dim=1)[0], dim=1))
    elif args.ood == "prob":
        if args.method == "max": scores = np.max(to_np(outputs), axis=1)
        if args.method == "sum": scores = np.sum(to_np(outputs), axis=1)
    elif args.ood == "msp":
        outputs = F.softmax(logits, dim=1)
        scores = np.max(to_np(outputs), axis=1)
    else:
        scores = logits_np

    return scores

def get_action_yolood(ood_heads, action, ood, n_classes, heads_to_use):
    obj1, cls1 = ood_heads[0].split((1, n_classes), -1)
    obj2, cls2 = ood_heads[1].split((1, n_classes), -1)
    obj3, cls3 = ood_heads[2].split((1, n_classes), -1)

    flat_and_max = lambda x: x.flatten(1, 2).max(dim=1)

    if action == 'cls':
        pred1, pred2, pred3 = flat_and_max(cls1)[0], flat_and_max(cls2)[0], flat_and_max(cls3)[0]
    elif action == 'obj':
        pred1, pred2, pred3 = flat_and_max(obj1)[0], flat_and_max(obj2)[0], flat_and_max(obj3)[0]
    else:  # action == 'obj*cls
        activation_fn = lambda x: x.sigmoid() if ood == 'yolood' else torch.log(1 + torch.exp(x))
        head1_idx = flat_and_max(activation_fn(obj1) * activation_fn(cls1))[1].flatten()
        head2_idx = flat_and_max(activation_fn(obj2) * activation_fn(cls2))[1].flatten()
        head3_idx = flat_and_max(activation_fn(obj3) * activation_fn(cls3))[1].flatten()

        batch_size = obj1.shape[0]
        b = torch.arange(0, batch_size).repeat(n_classes, 1).transpose(0, 1).flatten()
        c = torch.arange(0, n_classes).repeat(batch_size, 1).flatten()
        pred1 = torch.cat([obj1.flatten(1, 2)[b, head1_idx, 0],
                           cls1.flatten(1, 2)[b, head1_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)
        pred2 = torch.cat([obj2.flatten(1, 2)[b, head2_idx, 0],
                           cls2.flatten(1, 2)[b, head2_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)
        pred3 = torch.cat([obj3.flatten(1, 2)[b, head3_idx, 0],
                           cls3.flatten(1, 2)[b, head3_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)

    return torch.stack((pred1, pred2, pred3), -1)[..., heads_to_use]


def get_action_yolo(ood_heads, action, n_classes):
    _, obj_cls1 = ood_heads[0].split((4, 1 + n_classes), -1)
    _, obj_cls2 = ood_heads[1].split((4, 1 + n_classes), -1)
    _, obj_cls3 = ood_heads[2].split((4, 1 + n_classes), -1)
    obj1, cls1 = obj_cls1.split((1, n_classes), -1)
    obj2, cls2 = obj_cls2.split((1, n_classes), -1)
    obj3, cls3 = obj_cls3.split((1, n_classes), -1)

    flat_and_max = lambda x: x.flatten(1, 3).max(dim=1)

    if action == 'cls':
        pred1, pred2, pred3 = flat_and_max(cls1)[0], flat_and_max(cls2)[0], flat_and_max(cls3)[0]
    elif action == 'obj':
        pred1, pred2, pred3 = flat_and_max(obj1)[0], flat_and_max(obj2)[0], flat_and_max(obj3)[0]
    else:  # action == 'obj*cls
        head1_idx = flat_and_max(obj1.sigmoid() * cls1.sigmoid())[1].flatten()
        head2_idx = flat_and_max(obj2.sigmoid() * cls2.sigmoid())[1].flatten()
        head3_idx = flat_and_max(obj3.sigmoid() * cls3.sigmoid())[1].flatten()

        batch_size = obj1.shape[0]
        b = torch.arange(0, batch_size).repeat(n_classes, 1).transpose(0, 1).flatten()
        c = torch.arange(0, n_classes).repeat(batch_size, 1).flatten()
        pred1 = torch.cat([obj1.flatten(1, 3)[b, head1_idx, 0],
                           cls1.flatten(1, 3)[b, head1_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)
        pred2 = torch.cat([obj2.flatten(1, 3)[b, head2_idx, 0],
                           cls2.flatten(1, 3)[b, head2_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)
        pred3 = torch.cat([obj3.flatten(1, 3)[b, head3_idx, 0],
                           cls3.flatten(1, 3)[b, head3_idx, c]], -1).view(2, batch_size, n_classes).permute(1, 2, 0)

    return torch.stack((pred1, pred2, pred3), -1)


@torch.no_grad()
def get_yolood_scores(loader, model, args):
    """
    function that predict the model scores.
    the for loop stack all the model outputs after they perform the function get_action.
    after the for loop the function compute the score depend on which args.ood and args.ood_type chosen
    @param loader: train or test data loader
    @param model: the model
    @param args:
    @return: score - np.array in shape(loader size , 1)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    model.eval()
    n_classes = loader.dataset.n_classes
    for batch_idx, (images, targets) in enumerate(tqdm(loader)):
        images = images.to(device)
        output = model(images)
        pred = get_action_yolood(output[1], args.ood_type, args.ood, n_classes, args.use_heads)
        predictions.append(pred)

    predictions_t = torch.cat(predictions, dim=0)

    if args.ood == 'yolood':
        predictions_t = predictions_t.sigmoid()
    elif args.ood == 'energy':
        predictions_t = torch.log(1 + torch.exp(predictions_t))
    elif args.ood == 'logit':
        predictions_t = predictions_t
    elif args.ood == 'msp':
        predictions_t = F.softmax(predictions_t, dim=1)

    # [1, 2, N ,3]
    if args.ood_type == 'obj*cls':
        predictions_t = predictions_t[..., 0, :] * predictions_t[..., 1, :]

    # [1, N ,3]
    if args.head_method == 'max':
        predictions_t = predictions_t.max(dim=-1)[0]
    elif args.head_method == 'sum':
        sum_weights = torch.tensor([args.sum_weights[i] for i in args.use_heads], device=device)
        predictions_t = torch.matmul(predictions_t, sum_weights)
    elif args.head_method == 'multiply':
        predictions = torch.ones(predictions_t.shape[:-1], device=predictions_t.device)
        for i in range(len(args.use_heads)):
            predictions *= predictions_t[..., i]
        predictions_t = predictions

    if args.method == 'max':
        scores = predictions_t.max(dim=1)[0]
    elif args.method == 'sum':
        scores = predictions_t.sum(dim=1)

    return scores.cpu().numpy()


@torch.no_grad()
def get_yolo_scores(loader, model, args):
    """
    function that predict the model scores.
    the for loop stack all the model outputs after they perform the function get_action.
    after the for loop the function compute the score depend on which args.ood and args.ood_type chosen
    @param loader: train or test data loader
    @param model: the model
    @param args:
    @return: score - np.array in shape(loader size , 1)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    model.eval()
    n_classes = loader.dataset.n_classes
    for batch_idx, (images, targets) in enumerate(tqdm(loader)):
        images = images.to(device)
        output = model(images)
        pred = get_action_yolo(output[1], args.ood_type, n_classes)
        predictions.append(pred)

    predictions_t = torch.cat(predictions, dim=0)
    predictions_t = predictions_t.sigmoid()

    if args.ood_type == 'obj*cls':
        predictions_t = predictions_t[..., 0, :] * predictions_t[..., 1, :]

    if args.head_method == 'max':
        predictions_t = predictions_t.max(dim=-1)[0]
    elif args.head_method == 'sum':
        sum_weights = torch.tensor(args.sum_weights, device=device)
        predictions_t = torch.matmul(predictions_t, sum_weights)
    elif args.head_method == 'multiply':
        predictions_t = predictions_t[..., 0] * predictions_t[..., 1] * predictions_t[..., 2]

    if args.method == 'max':
        scores = predictions_t.max(dim=1)[0]
    elif args.method == 'sum':
        scores = predictions_t.sum(dim=1)

    return scores.cpu().numpy()


def get_maha_data(model, args):
    print('maha type: {}'.format(args.maha_type), flush=True)
    in_train_loader, _, _ = get_loaders(args, dataset_root=args.in_data_root, splits_to_load=['train'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Feature Extraction
    temp_x = torch.rand(1, 3, args.img_size, args.img_size).to(device)
    num_exits = len(model_feature_list(model, temp_x, args.arch))

    print('get sample mean and covariance', flush=True)
    sample_mean, precision = sample_estimator(model, in_train_loader,
                                              num_exits if args.maha_type == 'ensemble' else 1, args)
    return sample_mean, precision
