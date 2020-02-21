
import argparse

# Change 1
import horovod.tensorflow as hvd
# Change 2
hvd.init() 
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset




def main(args):

 

    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 32  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 393530798900


    '''
     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    '''
    options = {
     'bidirectional': True,
    
     'dropout': 0.15,
    
     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 100,
     'n_negative_samples_batch': 18,

     'optimizer_type': 'lamb',
     'learning_rate':0.0001,
     'warm_up_ratio':0.000005,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True, world_size=hvd.size(),global_rank=hvd.rank())

    # Change 3
    #args.save_dir = args.save_dir if hvd.rank() == 0 else os.path.join(args.save_dir, str(hvd.rank()))
    if hvd.rank() == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, hvd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--resume_ckpt_file', help='load last checkpoint')
    args = parser.parse_args()
    main(args)

