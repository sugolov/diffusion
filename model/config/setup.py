from utils.utils import save_config, load_config


CIFAR10_unet_config = {
    "layer_channels": [3, 8, 16, 32, 64],
    "layer_attention": [False, True, False, False],
    "layer_groups": (1, 2, 4, 8),
    "time_emb_dim": 1024,
    "time_n": 1e5,
    "kernel_size": 2,
    "upsample_size": 2,
    "residual": True,
    "padding": "same",
    "mlp_layers": None,
    "num_heads": 4,
    "embed_dim": 64,
    "out_groups": 4
}

if __name__ == "__main__":
    save_config(CIFAR10_unet_config, "CIFAR10_unet_config")
    config = load_config("CIFAR10_unet_config")