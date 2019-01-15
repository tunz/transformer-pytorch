
def prepare(problem_set, data_dir, max_length, batch_size, device, opt):
    if problem_set not in ['wmt32k', 'lm1b']:
        raise Exception("only ['wmt32k', 'lm1b'] problem set supported.")

    setattr(opt, 'share_target_embedding', False)
    setattr(opt, 'has_inputs', True)

    if problem_set == 'wmt32k':
        from dataset import translation
        train_iter, val_iter, opt = \
            translation.prepare(max_length, batch_size, device, opt, data_dir)
    elif problem_set == 'lm1b':
        from dataset import lm
        train_iter, val_iter, opt = \
            lm.prepare(max_length, batch_size, device, opt, data_dir)

    return train_iter, val_iter, opt.src_vocab_size, opt.trg_vocab_size, opt
