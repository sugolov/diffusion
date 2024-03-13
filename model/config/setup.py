from utils.utils import save_config

CIFAR10_unet_config = {
    "layer_channels": [3, 8, 16, 32, 64],
    "layer_attention": [False, True, False, False],
    "time_emb_dim": 1024,
    "time_n": 1e5,
    "kernel_size": 2,
    "upsample_size": 2,
    "residual": True,
    "padding": "same",
    "num_groups": 4,
    "mlp_layers": (1024,),
    "num_heads": 4,
    "embed_dim": 16
}

if __name__ == "__main__":
    save_config(CIFAR10_unet_config, "CIFAR10_unet_config")