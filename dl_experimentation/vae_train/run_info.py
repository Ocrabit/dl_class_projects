# dainty-sweep-158 (cjugxj69)
config_dainty_sweep_158 = dict(
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
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

# silver-sweep-152 (yd9u1vxj)
config_silver_sweep_152 = dict(
    latent_shape=(1, 7, 7),
    base_channels=32,
    blocks_per_level=2,
    groups=1,
    dropout=0.235134,
    activation='SiLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.000764,
    weight_decay=0.000085,
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

# stoic-sweep-74 (k8btel7f)
config_stoic_sweep_74 = dict(
    latent_shape=(1, 7, 7),
    base_channels=32,
    blocks_per_level=3,
    groups=1,
    dropout=0.218544,
    activation='GELU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.000595,
    weight_decay=0.000041,
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

# young-sweep-114 (9f4vb0hg)
config_young_sweep_114 = dict(
    latent_shape=(1, 7, 7),
    base_channels=24,
    blocks_per_level=1,
    groups=1,
    dropout=0.349223,
    activation='ReLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.003509,
    weight_decay=0.000089,
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

# earthy-sweep-151 (bzkr37oe)
config_earthy_sweep_151 = dict(
    latent_shape=(1, 7, 7),
    base_channels=24,
    blocks_per_level=1,
    groups=4,
    dropout=0.327277,
    activation='GELU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.009207,
    weight_decay=0.000002,
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)

# glowing-sweep-110 (zf81gxm4)
config_glowing_sweep_110 = dict(
    latent_shape=(1, 7, 7),
    base_channels=16,
    blocks_per_level=3,
    groups=4,
    dropout=0.432183,
    activation='ReLU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=0.007816,
    weight_decay=0.000016,
    epochs=25,

    beta_final=0.105,
    warmup_epochs=5,
    ema=0.97,
)