import hydra
from omegaconf import DictConfig

from trainer.training_pipeline import train


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config: DictConfig):
    # Train model
    return train(config)

if __name__ == "__main__":
    main()
