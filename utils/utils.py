import shutil
import math

import torch
import torch.nn.functional as F


def get_loss(pred, ans, vocab_size, label_smoothing, pad):
    # took this "normalizing" from tensor2tensor. We subtract it for
    # readability. This makes no difference on learning.
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    normalizing = -(
        confidence * math.log(confidence) + float(vocab_size - 1) *
        low_confidence * math.log(low_confidence + 1e-20))

    one_hot = torch.zeros_like(pred).scatter_(1, ans.unsqueeze(1), 1)
    one_hot = one_hot * confidence + (1 - one_hot) * low_confidence
    log_prob = F.log_softmax(pred, dim=1)

    xent = -(one_hot * log_prob).sum(dim=1)
    xent = xent.masked_select(ans != pad)
    loss = (xent - normalizing).mean()
    return loss


def get_accuracy(pred, ans, pad):
    pred = pred.max(1)[1]
    n_correct = pred.eq(ans)
    n_correct = n_correct.masked_select(ans != pad).sum().item()
    return n_correct / ans.size(0)


def save_checkpoint(model, filepath, global_step, is_best):
    model_save_path = filepath + '/last_model.pt'
    torch.save(model, model_save_path)
    torch.save(global_step, filepath + '/global_step.pt')
    if is_best:
        best_save_path = filepath + '/best_model.pt'
        shutil.copyfile(model_save_path, best_save_path)


def load_checkpoint(model_path, device, is_eval=True):
    if is_eval:
        model = torch.load(model_path + '/best_model.pt')
        model.eval()
        return model.to(device=device)

    model = torch.load(model_path + '/last_model.pt')
    global_step = torch.load(model_path + '/global_step.pt')
    return model.to(device=device), global_step


def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(targets):
    # Prevent leftward information flow in self-attention.
    target_len = targets.size()[1]
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=targets.device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask
