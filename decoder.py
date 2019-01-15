import argparse
import time

import torch
import torch.nn.functional as F

from utils import utils

# pylint: disable=not-callable


def encode_inputs(sentence, model, src_data, beam_size, device):
    inputs = src_data['field'].preprocess(sentence)
    inputs.append(src_data['field'].eos_token)
    inputs = [inputs]
    inputs = src_data['field'].process(inputs, device=device)
    with torch.no_grad():
        src_mask = utils.create_pad_mask(inputs, src_data['pad_idx'])
        enc_output = model.encode(inputs, src_mask)
        enc_output = enc_output.repeat(beam_size, 1, 1)
    return enc_output, src_mask


def update_targets(targets, best_indices, idx, vocab_size):
    best_tensor_indices = torch.div(best_indices, vocab_size)
    best_token_indices = torch.fmod(best_indices, vocab_size)
    new_batch = torch.index_select(targets, 0, best_tensor_indices)
    new_batch[:, idx] = best_token_indices
    return new_batch


def get_result_sentence(indices_history, trg_data, vocab_size):
    result = []
    k = 0
    for best_indices in indices_history[::-1]:
        best_idx = best_indices[k]
        # TODO: get this vocab_size from target.pt?
        k = best_idx // vocab_size
        best_token_idx = best_idx % vocab_size
        best_token = trg_data['field'].vocab.itos[best_token_idx]
        result.append(best_token)
    return ' '.join(result[::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--translate', action='store_true')
    args = parser.parse_args()

    beam_size = args.beam_size

    # Load fields.
    if args.translate:
        src_data = torch.load(args.data_dir + '/source.pt')
    trg_data = torch.load(args.data_dir + '/target.pt')

    # Load a saved model.
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    model = utils.load_checkpoint(args.model_dir, device)

    pads = torch.tensor([trg_data['pad_idx']] * beam_size, device=device)
    pads = pads.unsqueeze(-1)

    # We'll find a target sequence by beam search.
    scores_history = [torch.zeros((beam_size,), dtype=torch.float,
                                  device=device)]
    indices_history = []
    cache = {}

    eos_idx = trg_data['field'].vocab.stoi[trg_data['field'].eos_token]

    if args.translate:
        sentence = input('Source? ')

    # Encoding inputs.
    if args.translate:
        start_time = time.time()
        enc_output, src_mask = encode_inputs(sentence, model, src_data,
                                             beam_size, device)
        targets = pads
        start_idx = 0
    else:
        enc_output, src_mask = None, None
        sentence = input('Target? ').split()
        for idx, _ in enumerate(sentence):
            sentence[idx] = trg_data['field'].vocab.stoi[sentence[idx]]
        sentence.append(trg_data['pad_idx'])
        targets = torch.tensor([sentence], device=device)
        start_idx = targets.size(1) - 1
        start_time = time.time()

    with torch.no_grad():
        for idx in range(start_idx, args.max_length):
            if idx > start_idx:
                targets = torch.cat((targets, pads), dim=1)
            t_self_mask = utils.create_trg_self_mask(targets)

            t_mask = utils.create_pad_mask(targets, trg_data['pad_idx'])
            pred = model.decode(targets, enc_output, src_mask,
                                t_self_mask, t_mask, cache)
            pred = pred[:, idx].squeeze(1)
            vocab_size = pred.size(1)

            pred = F.log_softmax(pred, dim=1)
            if idx == start_idx:
                scores = pred[0]
            else:
                scores = scores_history[-1].unsqueeze(1) + pred
            length_penalty = pow(((5. + idx + 1.) / 6.), args.alpha)
            scores = scores / length_penalty
            scores = scores.view(-1)

            best_scores, best_indices = scores.topk(beam_size, 0)
            scores_history.append(best_scores)
            indices_history.append(best_indices)

            # Stop searching when the best output of beam is EOS.
            if best_indices[0].item() % vocab_size == eos_idx:
                break

            targets = update_targets(targets, best_indices, idx, vocab_size)

    result = get_result_sentence(indices_history, trg_data, vocab_size)
    print("Result: {}".format(result))

    print("Elapsed Time: {:.2f} sec".format(time.time() - start_time))


if __name__ == '__main__':
    main()
