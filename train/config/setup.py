from utils.utils import save_config, load_config

CIFAR10_train_config = {
    "train": {
        "lr": 2e-4,
        "epochs": 8e5,
        "batch_size": 128,
        "checkpoint_steps": 10
    },
    "ddpm": {
        "n_steps": 1000,
        "noise_schedule_start": 1e-4,
        "noise_schedule_end": 2e-2,
    }
}

if __name__ == "__main__":
    save_config(CIFAR10_train_config, "CIFAR10_train_config")
    config = load_config("CIFAR10_train_config")