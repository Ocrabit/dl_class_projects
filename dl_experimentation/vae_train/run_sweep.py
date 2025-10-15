import sys
from train import train
from run_info import (
    config_dainty_sweep_158,
    config_silver_sweep_152,
    config_stoic_sweep_74,
    config_young_sweep_114,
    config_earthy_sweep_151,
    config_glowing_sweep_110,
)

CONFIGS = {
    "dainty": config_dainty_sweep_158,
    "silver": config_silver_sweep_152,
    "stoic": config_stoic_sweep_74,
    "young": config_young_sweep_114,
    "earthy": config_earthy_sweep_151,
    "glowing": config_glowing_sweep_110,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_sweep.py <config_name> [project_name]")
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else "vae_conv_testing"

    if config_name not in CONFIGS:
        print(f"Error: Unknown config '{config_name}'")
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    config = CONFIGS[config_name]
    print(f"Running training with config: {config_name}")
    train(config=config, project=project_name)