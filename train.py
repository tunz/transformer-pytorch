import argparse
import os
import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import problem
from utils.optimizer import LRScheduler
from utils import utils


def summarize_train(writer, global_step, last_time, model, opt,
                    inputs, targets, optimizer, loss, pred, ans):
    if opt.summary_grad:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            norm = torch.norm(param.grad.data.view(-1))
            writer.add_scalar('gradient_norm/' + name, norm,
                              global_step)

    writer.add_scalar('input_stats/batch_size',
                      targets.size(0), global_step)

    if inputs is not None:
        writer.add_scalar('input_stats/input_length',
                          inputs.size(1), global_step)
        i_nonpad = (inputs != opt.src_pad_idx).view(-1).type(torch.float32)
        writer.add_scalar('input_stats/inputs_nonpadding_frac',
                          i_nonpad.mean(), global_step)

    writer.add_scalar('input_stats/target_length',
                      targets.size(1), global_step)
    t_nonpad = (targets != opt.trg_pad_idx).view(-1).type(torch.float32)
    writer.add_scalar('input_stats/target_nonpadding_frac',
                      t_nonpad.mean(), global_step)

    writer.add_scalar('optimizer/learning_rate',
                      optimizer.learning_rate(), global_step)

    writer.add_scalar('loss', loss.item(), global_step)

    acc = utils.get_accuracy(pred, ans, opt.trg_pad_idx)
    writer.add_scalar('training/accuracy',
                      acc, global_step)

    steps_per_sec = 100.0 / (time.time() - last_time)
    writer.add_scalar('global_step/sec', steps_per_sec,
                      global_step)


def train(train_data, model, opt, global_step, optimizer, t_vocab_size,
          label_smoothing, writer):
    model.train()
    last_time = time.time()
    pbar = tqdm(total=len(train_data.dataset), ascii=True)
    for batch in train_data:
        inputs = None
        if opt.has_inputs:
            inputs = batch.src

        targets = batch.trg
        pred = model(inputs, targets)

        pred = pred.view(-1, pred.size(-1))
        ans = targets.view(-1)

        loss = utils.get_loss(pred, ans, t_vocab_size,
                              label_smoothing, opt.trg_pad_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            summarize_train(writer, global_step, last_time, model, opt,
                            inputs, targets, optimizer, loss, pred, ans)
            last_time = time.time()

        pbar.set_description('[Loss: {:.4f}]'.format(loss.item()))

        global_step += 1
        pbar.update(targets.size(0))

    pbar.close()
    train_data.reload_examples()
    return global_step


def validation(validation_data, model, global_step, t_vocab_size, val_writer,
               opt):
    model.eval()
    total_loss = 0.0
    total_cnt = 0
    for batch in validation_data:
        inputs = None
        if opt.has_inputs:
            inputs = batch.src
        targets = batch.trg

        with torch.no_grad():
            pred = model(inputs, targets)

            pred = pred.view(-1, pred.size(-1))
            ans = targets.view(-1)
            loss = utils.get_loss(pred, ans, t_vocab_size, 0,
                                  opt.trg_pad_idx)
        total_loss += loss.item() * len(batch)
        total_cnt += len(batch)

    val_loss = total_loss / total_cnt
    print("Validation Loss", val_loss)
    val_writer.add_scalar('loss', val_loss, global_step)
    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', required=True)
    parser.add_argument('--train_step', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--filter_size', type=int, default=2048)
    parser.add_argument('--warmup', type=int, default=16000)
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--summary_grad', action='store_true')
    opt = parser.parse_args()

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    if not os.path.exists(opt.output_dir + '/last/models'):
        os.makedirs(opt.output_dir + '/last/models')
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)

    train_data, validation_data, i_vocab_size, t_vocab_size, opt = \
        problem.prepare(opt.problem, opt.data_dir, opt.max_length,
                        opt.batch_size, device, opt)
    if i_vocab_size is not None:
        print("# of vocabs (input):", i_vocab_size)
    print("# of vocabs (target):", t_vocab_size)

    if opt.model == 'transformer':
        from model.transformer import Transformer
        model_fn = Transformer
    elif opt.model == 'fast_transformer':
        from model.fast_transformer import FastTransformer
        model_fn = FastTransformer

    if os.path.exists(opt.output_dir + '/last/models/last_model.pt'):
        print("Load a checkpoint...")
        last_model_path = opt.output_dir + '/last/models'
        model, global_step = utils.load_checkpoint(last_model_path, device,
                                                   is_eval=False)
    else:
        model = model_fn(i_vocab_size, t_vocab_size,
                         n_layers=opt.n_layers,
                         hidden_size=opt.hidden_size,
                         filter_size=opt.filter_size,
                         dropout_rate=opt.dropout,
                         share_target_embedding=opt.share_target_embedding,
                         has_inputs=opt.has_inputs,
                         src_pad_idx=opt.src_pad_idx,
                         trg_pad_idx=opt.trg_pad_idx)
        model = model.to(device=device)
        global_step = 0

    if opt.parallel:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of parameters: {}".format(num_params))

    optimizer = LRScheduler(
        filter(lambda x: x.requires_grad, model.parameters()),
        opt.hidden_size, opt.warmup, step=global_step)

    writer = SummaryWriter(opt.output_dir + '/last')
    val_writer = SummaryWriter(opt.output_dir + '/last/val')
    best_val_loss = float('inf')

    for t_step in range(opt.train_step):
        print("Epoch", t_step)
        start_epoch_time = time.time()
        global_step = train(train_data, model, opt, global_step,
                            optimizer, t_vocab_size, opt.label_smoothing,
                            writer)
        print("Epoch Time: {:.2f} sec".format(time.time() - start_epoch_time))

        if t_step % opt.val_every != 0:
            continue

        val_loss = validation(validation_data, model, global_step,
                              t_vocab_size, val_writer, opt)
        utils.save_checkpoint(model, opt.output_dir + '/last/models',
                              global_step, val_loss < best_val_loss)
        best_val_loss = min(val_loss, best_val_loss)


if __name__ == '__main__':
    main()
