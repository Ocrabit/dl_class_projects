from train import train
import sys

config_final_spatial = dict(
    latent_shape=(1, 7, 7),
    base_channels=32,
    blocks_per_level=1,
    groups=4,
    dropout=0.294905,
    activation='SiLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.004303,
    weight_decay=0.000039,
    epochs=2000,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

config_final_flat = dict(
    latent_shape=49,
    base_channels=32,
    blocks_per_level=1,
    groups=4,
    dropout=0.294905,
    activation='SiLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.004303,
    weight_decay=0.000039,
    epochs=2000,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

config_small_sad_competition = dict(
    latent_shape=16,
    base_channels=16,
    blocks_per_level=2,
    groups=4,
    dropout=0.3,
    activation='GeLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.005,
    weight_decay=0.00004,
    epochs=200,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Mode not specified. Use 'spatial' or 'flat'")
        sys.exit(1)

    mode = sys.argv[1].lower()
    project = "final_6_vae"

    if mode == 'spatial':
        print("Running spatial config (1,7,7)")
        checkpoint_spatial = r'..\checkpoints\vae\bright-sun-14\current.pt'
        train(config=config_final_spatial, project=project, checkpoint_path=checkpoint_spatial)
    elif mode == 'flat':
        checkpoint_flat = r'..\checkpoints\vae\rich-hill-15\current.pt'
        print("Running flat config (49)")
        train(config=config_final_flat, project=project, checkpoint_path=checkpoint_flat)
    elif mode == 'small-sad-competition':
        # checkpoint_flat = r'..\checkpoints\vae\rich-hill-15\current.pt'
        print("Running small-sad-competition config (16)")
        train(config=config_small_sad_competition, project=project)
    else:
        print(f"Unknown mode: {mode}. Use 'spatial' or 'flat'")
        sys.exit(1)
