import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import os

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================================
# 这个脚本做什么？
# --------------------------------------------------------------------------------------
# 目标：从零实现并训练一个最基础的 Transformer（Decoder-only / GPT 结构）
# 任务：字符级语言建模（Character-level Language Modeling）
#
# 输入（训练时）:
#   - x: [B, T] 的 token id（整数）
# 输出（训练时）:
#   - logits: [B, T, V] 的未归一化打分
# 监督信号:
#   - y: [B, T]，是 x 整体右移一位后的“下一个字符”
# 损失:
#   - cross entropy
# 推理/生成:
#   - 使用 generate() 自回归采样生成更多字符
# ======================================================================================


# ----------------------------
# 1) 配置区：超参数集中管理
# ----------------------------
@dataclass
class Config:
    # 数据
    data_path: str = "input.txt"
    train_split: float = 0.9

    # 训练
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # 模型
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    # 生成
    gen_len: int = 300
    seed: int = 1337

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

# ----------------------------
# W&B 初始化
# ----------------------------
wandb_run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "mini-gpt-char"),
    entity=os.getenv("WANDB_ENTITY", None),  # 建议用环境变量 WANDB_ENTITY
    name=os.getenv("WANDB_RUN_NAME", None),
    config=asdict(cfg),
)

# 固定随机种子（影响：权重初始化、batch 随机采样、生成采样等）
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

# ----------------------------
# CKPT 保存设置
# ----------------------------
save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

ckpt_best_path = save_dir / "ckpt_best.pt"
ckpt_final_path = save_dir / "ckpt_final.pt"
best_val = float("inf")


# ----------------------------
# 2) 读取文本并构建字符级词表
# ----------------------------
with open(cfg.data_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    """字符串 -> token id 列表（字符级）"""
    return [stoi[c] for c in s]


def decode(ids):
    """token id 列表 -> 字符串"""
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(cfg.train_split * len(data))
train_data = data[:n]
val_data = data[n:]


# ----------------------------
# 3) 构造 batch：随机切片做 next-token 监督
# ----------------------------
def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix = torch.randint(0, len(src) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def estimate_loss(model: nn.Module):
    """
    在 train 与 val 上估计 loss，用于监控训练是否过拟合/是否收敛。
    - no_grad: 不记录梯度
    - eval():  关闭 dropout 等随机层
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters, device=cfg.device)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


# ----------------------------
# 4) Transformer 组件：注意力 / FFN / Block
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 一次性线性变换得到 Q/K/V（参数在这里：Wq/Wk/Wv 的合并形式）
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # 多头合并后的输出投影 Wo
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask：下三角为 1，上三角为 0
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        qkv = self.qkv(x)           # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)  # 各 [B, T, C]

        # [B, T, C] -> [B, nh, T, hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力分数: [B, nh, T, T]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask：禁止看未来
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # softmax -> 权重
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 加权求和 -> [B, nh, T, hd]
        y = att @ v

        # 合并多头 -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + dropout
        y = self.resid_drop(self.proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-LN（更稳定）：LN -> 子层 -> 残差
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ----------------------------
# 5) 最小 GPT：Embedding + N个Block + LM Head
# ----------------------------
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape
        assert T <= self.cfg.block_size, "序列长度不能超过 block_size"

        pos = torch.arange(0, T, device=idx.device)  # [T]

        # [B, T] -> [B, T, C]
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        x = self.blocks(x)
        x = self.ln_f(x)

        # [B, T, C] -> [B, T, V]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size),
                targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """
        自回归生成：
          - 每次取最后一个位置的 logits -> softmax -> 采样 -> 拼接
          - idx 会越来越长，但每次喂给模型的最多是最后 block_size 个 token
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # [B, V]

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)               # [B, V]
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_id], dim=1)          # [B, T+1]
        return idx


# ----------------------------
# 构建模型与优化器
# ----------------------------
model = GPTLanguageModel(vocab_size, cfg).to(cfg.device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay
)


def save_ckpt(path: Path, it: int, best_val_loss: float | None = None):
    """
    保存 checkpoint（用于后续加载推理/断点复现）。
    关键：字符级模型必须保存 stoi/itos，否则无法 encode/decode。
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "iter": it,
    }
    if best_val_loss is not None:
        ckpt["best_val"] = best_val_loss
    torch.save(ckpt, str(path))
    print(f"[ckpt] saved to {path} (iter={it})")


# ----------------------------
# 6) 训练主程序（带 best ckpt 保存）
# ----------------------------
t0 = time.time()
for it in range(1, cfg.max_iters + 1):

    # 定期评估
    if it % cfg.eval_interval == 0 or it == 1:
        losses = estimate_loss(model)
        dt = time.time() - t0
        print(
            f"iter {it:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"{dt:.1f}s"
        )

        wandb.log(
            {
                "eval/train_loss": losses["train"],
                "eval/val_loss": losses["val"],
            },
            step=it
        )

        # 保存 best（按 val loss 最小）
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_ckpt(ckpt_best_path, it=it, best_val_loss=best_val)

            # 可选：上传到 W&B（artifact）
            artifact = wandb.Artifact(name="ckpt_best", type="model")
            artifact.add_file(str(ckpt_best_path))
            wandb_run.log_artifact(artifact)

        t0 = time.time()

    # 训练一步
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    wandb.log(
        {
            "train/loss_step": float(loss.item()),
            "train/iter": it,
            "train/lr": float(optimizer.param_groups[0]["lr"]),
        },
        step=it
    )

# ----------------------------
# 训练结束：保存 final checkpoint
# ----------------------------
save_ckpt(ckpt_final_path, it=cfg.max_iters)

artifact = wandb.Artifact(name="ckpt_final", type="model")
artifact.add_file(str(ckpt_final_path))
wandb_run.log_artifact(artifact)

# ----------------------------
# 7) 训练结束：做一次生成看看模型学到什么
# ----------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
out = model.generate(context, cfg.gen_len, temperature=0.8, top_k=50)[0].tolist()

print("\n=== sample ===")
sample_text = decode(out)
print(sample_text)

wandb.log({"sample/text": sample_text})
wandb.finish()
