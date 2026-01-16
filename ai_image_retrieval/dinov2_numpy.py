import numpy as np
from scipy.ndimage import zoom
from PIL import Image

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size = 14

        self.cls_token           = weights["embeddings.cls_token"]  # (1, 1, 768)
        self.position_embeddings = weights["embeddings.position_embeddings"]  # (1, 1370, 768) for base
        # patch projection: weight 原 shape (768, 3, 14, 14) -> flatten 并转置为 (3*14*14, 768)
        proj_w                   = weights["embeddings.patch_embeddings.projection.weight"]
        self.patch_embed_w       = proj_w.reshape(768, -1).T  # (3*14*14, 768)
        self.patch_embed_b       = weights["embeddings.patch_embeddings.projection.bias"][None, :]  # (1, 768)

    def pixel2patches(self, pixel_values):  # (B, 3, H, W) -> (B, num_patches, 3*14*14)
        B, C, H, W = pixel_values.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, f"Input size ({H}, {W}) must be divisible by patch_size {ps}"

        # 高效无循环提取 patches
        patches = pixel_values.reshape(B, C, H // ps, ps, W // ps, ps)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)  # (B, h, w, C, ps, ps)
        patches = patches.reshape(B, -1, C * ps * ps)  # (B, h*w, 3*14*14)
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        B, seq_len, D = embeddings.shape
        num_patches = seq_len - 1
        ps = self.patch_size
        new_h, new_w = height // ps, width // ps
        assert num_patches == new_h * new_w

        # 原始位置编码 (1, 1370, 768)
        pos_embed = self.position_embeddings[0]  # (1370, 768)
        cls_pos = pos_embed[0:1, :]  # (1, 768)
        patch_pos = pos_embed[1:, :]  # (1369, 768)

        # DINOv2 base 训练于 518x518 -> 37x37 patches
        orig_grid = 37
        patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, D)

        # 双线性插值到新尺寸
        scale_h = new_h / orig_grid
        scale_w = new_w / orig_grid
        interpolated = zoom(patch_pos, (1, scale_h, scale_w, 1), order=1)  # (1, new_h, new_w, D)

        interpolated = interpolated.reshape(1, new_h * new_w, D)
        new_pos = np.concatenate([cls_pos[None, :, :], interpolated], axis=1)  # (1, seq_len, D)

        # tile 到 batch
        new_pos = np.tile(new_pos, (B, 1, 1))  # (B, seq_len, D)
        return new_pos

    def __call__(self, pixel_values):
        B, C, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values)  # (B, num_patches, 3*14*14)

        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b  # (B, num_patches, 768)

        cls_token = np.tile(self.cls_token, (B, 1, 1))  # (B, 1, 768)
        embeddings = np.concatenate([cls_token, embeddings], axis=1)  # (B, seq_len, 768)

        pos_embed = self.interpolate_pos_encoding(embeddings, H, W)
        embeddings = embeddings + pos_embed
        return embeddings

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return norm * self.weight + self.bias

class LayerScale:
    def __init__(self, lambda1):
        self.lambda1 = lambda1

    def __call__(self, x):
        return x * self.lambda1

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight  # torch format: (out_features, in_features)
        self.bias = bias

    def __call__(self, x):
        return x @ self.weight.T + self.bias

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5  # 注意：这里除以 sqrt(head_dim)，不是 sqrt(hidden_size)

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        B, N, D = x.shape

        # (B, N, D) -> (B, N, h, d) -> (B, h, N, d)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 关键修复：k 需要转置成 (B, h, d, N)
        # 原 transpose(0, 2, 3, 2) 有重复轴 2，导致报错
        # 正确：transpose(0, 1, 3, 2)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, h, N, N)
        attn = softmax(attn, axis=-1)

        out = attn @ v  # (B, h, N, d)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)  # (B, N, h, d) -> (B, N, D)

        return self.out_proj(out)

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"
        
        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(f"{prefix}", weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.weights = weights
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "patch_size": 14,
        }

        self.embeddings = Embeddings(weights)
        self.blocks = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # (B, 768) cls token feature