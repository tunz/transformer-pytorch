
def prepare(problem_set, data_dir, max_length, batch_size, device, opt):
    if problem_set not in ['wmt32k']:
        raise Exception("only wmt32k problem set supported.")

    setattr(opt, 'share_target_embedding', False)

    if problem_set == 'wmt32k':
        from dataset import translation
        train_iter, val_iter, opt = \
            translation.prepare(max_length, batch_size, device, opt, data_dir)

    return train_iter, val_iter, opt.src_vocab_size, opt.trg_vocab_size, opt
