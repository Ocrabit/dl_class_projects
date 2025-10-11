import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--base_channels', type=int, required=True)
    parser.add_argument('--blocks_per_level', type=int, required=True)
    parser.add_argument('--groups', type=int, required=True)
    parser.add_argument('--dropout', type=float, required=True)
    parser.add_argument('--activation', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--beta_final', type=float, required=True)
    parser.add_argument('--warmup_epochs', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ema', type=float, default=0.97)
    parser.add_argument('--use_skips', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_bn', type=lambda x: x.lower() == 'true', default=True)

    args = parser.parse_args()
    train(config=vars(args), project="final_6_vae")