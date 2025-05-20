import wandb
from train import train

sweep_config = {
    'method': 'bayes',
    # 'name': 'translit-seq2seq-attention',
    'name': 'translit-seq2seq-tamil',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'emb_dim': {'values': [16,32,64,256]},
        'hid_dim': {'values': [16,32,64,256]},
        'enc_layers': {'values': [1,2,3]},
        # 'enc_layers': {'values': [1]},
        'dec_layers': {'values': [1,2,3]},
        'dec_layers': {'values': [1]},
        'cell_type': {'values': ['rnn','gru','lstm']},
        'dropout': {'values': [0.0,0.1]},
        'lr': {'values': [1e-2,1e-4]},
        'batch_size': {'values': [32,64]},
        'beam_size':   {'values': [1,5,10]},
        'max_len': { 'values': [32]},
        'epochs': {'values': [15,20]},
        'use_attention': {'values': [False]},
        # 'use_attention': {'values': [True]},
        'bidirectional': {'values': [True, False]},
        # 'bidirectional': {'values': [False]},
        'teacher_forcing_ratio':{'values': [0.0,0.2]},
    }
}

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='test-project')
    wandb.agent(sweep_id, function=lambda: train()) #count=100