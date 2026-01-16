import numpy as np
from scipy.ndimage import zoom

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight.T  # (in, out)
        self.bias = bias.reshape(1, -1)

    def __call__(self, x):
        return x @ self.weight + self.bias

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight.reshape(1, -1)
        self.bias = bias.reshape(1, -1)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.weight + self.bias

class LayerScale:
    def __init__(self, lambda1):
        self.lambda1 = lambda1.reshape(1, -1)

    def __call__(self, x):
        return x * self.lambda1

class MLP:
    def __init__(self, prefix, weights):
        self.fc1 = Linear(weights[f"{prefix}.mlp.fc1.weight"], weights[f"{prefix}.mlp.fc1.bias"])
        self.fc2 = Linear(weights[f"{prefix}.mlp.fc2.weight"], weights[f"{prefix}.mlp.fc2.bias"])

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(weights[f"{prefix}.attention.query.weight"], weights[f"{prefix}.attention.query.bias"])
        self.k_proj = Linear(weights[f"{prefix}.attention.key.weight"], weights[f"{prefix}.attention.key.bias"])
        self.v_proj = Linear(weights[f"{prefix}.attention.value.weight"], weights[f"{prefix}.attention.value.bias"])
        self.out_proj = Linear(weights[f"{prefix}.output.dense.weight"], weights[f"{prefix}.output.dense.bias"])

    def __call__(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 2, 3, 1)) * self.scale
        attn = softmax(attn, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size = 14

        self.cls_token = weights["embeddings.cls_token"]
        self.position_embeddings = weights["embeddings.position_embeddings"]

        w = weights["embeddings.patch_embeddings.projection.weight"]
        self.patch_embed_w = w.reshape(768, -1).T
        self.patch_embed_b = weights["embeddings.patch_embeddings.projection.bias"].reshape(1, 768)

    def pixel2patches(self, pixel_values):
        B, C, H, W = pixel_values.shape
        ps = self.patch_size
        patches = pixel_values.reshape(B, C, H//ps, ps, W//ps, ps)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(B, -1, C * ps * ps)

    def interpolate_pos_encoding(self, embeddings, height, width):
        B, num_tokens, D = embeddings.shape
        ps = self.patch_size

        pos_embed = self.position_embeddings
        old_N = pos_embed.shape[1] - 1
        grid_old = int(np.sqrt(old_N))

        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :].reshape(1, grid_old, grid_old, D)

        h_p = height // ps
        w_p = width // ps
        patch_pos = zoom(patch_pos, (1, h_p / grid_old, w_p / grid_old, 1), order=3)
        patch_pos = patch_pos.reshape(1, h_p * w_p, D)

        pos_embed = np.concatenate([cls_pos, patch_pos], axis=1)
        return np.tile(pos_embed, (B, 1, 1))

    def __call__(self, pixel_values):
        B, C, H, W = pixel_values.shape
        patches = self.pixel2patches(pixel_values)
        x = patches @ self.patch_embed_w + self.patch_embed_b

        cls = np.tile(self.cls_token, (B, 1, 1))
        x = np.concatenate([cls, x], axis=1)

        pos = self.interpolate_pos_encoding(x, H, W)
        return x + pos

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, prefix, weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(prefix, weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights_path='vit-dinov2-base.npz', config=None):
        # 支持 .npz 加载
        loaded = np.load(weights_path)
        weights = {k: loaded[k] for k in loaded.files}  # 转为普通 dict

        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks = [TransformerBlock(self.config, i, weights) for i in range(12)]
        self.norm = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # (B, 768)