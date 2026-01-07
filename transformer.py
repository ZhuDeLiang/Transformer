import math
import time
from dataclasses import dataclass
import os
import wandb
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================================
# 这个脚本做什么？
# --------------------------------------------------------------------------------------
# 目标：从零实现并训练一个最基础的 Transformer（Decoder-only / GPT 结构）
# 任务：字符级语言建模（Character-level Language Modeling）
#
# 输入（训练时）：
#   - x: 形状 [B, T] 的整数张量，每个元素是一个字符的 token id
#        B = batch_size（一次并行训练多少条序列）
#        T = block_size（每条序列的长度，也叫上下文长度）
#
# 输出（训练时）：
#   - logits: 形状 [B, T, V] 的浮点张量
#        V = vocab_size（字符种类数）
#        logits[b, t, :] 是对“位置 t 的下一个字符”的类别打分（未归一化概率）
#
# 监督信号（训练目标）：
#   - y: 形状 [B, T] 的整数张量，是 x 整体右移一位后的 token id
#        y[b, t] 是 x[b, t] 对应位置的“真实下一个字符”
#
# 损失：
#   - cross entropy：把每个位置的分类问题累加/平均
#
# 推理/生成（生成文本）：
#   - 输入一段上下文 idx: [B, T]
#   - 输出追加若干 token，最终得到更长序列 [B, T + max_new_tokens]
# ======================================================================================


# ----------------------------
# 1) 配置区：超参数集中管理
# ----------------------------
@dataclass
class Config:
    # 数据文件路径：一个纯文本文件，里面是训练语料（例如 Tiny Shakespeare）
    data_path: str = "input.txt"

    # 训练集/验证集划分比例（0.9 表示 90% 用于训练，10% 用于验证）
    train_split: float = 0.9

    # ------------------ 训练超参数 ------------------
    batch_size: int = 64
    # block_size = 上下文长度 T：
    # 模型每次看到 T 个字符，并对每个位置预测“下一个字符”
    block_size: int = 128

    # 最大训练迭代步数（不是 epoch，这里是 iterations）
    max_iters: int = 3000

    # 每隔多少步做一次评估（打印 train/val loss）
    eval_interval: int = 300

    # 评估时采样多少个 batch 取平均，降低抖动
    eval_iters: int = 100

    # AdamW 学习率
    learning_rate: float = 3e-4

    # AdamW 的权重衰减（L2 正则化的一种更合理实现）
    weight_decay: float = 1e-2

    # 梯度裁剪阈值（防止梯度爆炸；尤其是 Transformer 训练早期）
    grad_clip: float = 1.0

    # ------------------ 模型结构超参数 ------------------
    # n_embd = embedding 维度 C（隐藏状态维度）
    n_embd: int = 256

    # n_head = multi-head attention 的头数 nh
    n_head: int = 4

    # n_layer = Transformer Block 的层数（堆叠多少层）
    n_layer: int = 4

    # dropout 概率：训练时随机置零部分激活，减少过拟合
    dropout: float = 0.1

    # ------------------ 生成相关 ------------------
    # 训练结束后生成多少个新字符，用来直观看模型学到什么
    gen_len: int = 300

    # 随机种子：保证可复现（同样的代码+同样的数据+同样种子 -> 类似结果）
    seed: int = 1337

    # ------------------ 设备 ------------------
    # 有 GPU 用 cuda，没有则 cpu
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 实例化配置
cfg = Config()
wandb_run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "mini-gpt-char"),
    entity=os.getenv("572424159-fudan-university-school-of-management", None),
    name=os.getenv("WANDB_RUN_NAME", None),  # 可选：在环境变量里指定
    config=asdict(cfg),                      # 把 dataclass 配置记录到 W&B
)
# 固定随机种子（影响权重初始化、batch 随机采样、生成时采样等）
torch.manual_seed(cfg.seed)


# ----------------------------
# 2) 读取文本并构建字符级词表
# ----------------------------
# 读取原始文本语料（一个长字符串）
with open(cfg.data_path, "r", encoding="utf-8") as f:
    text = f.read()

# chars：语料里出现过的所有不同字符（去重后排序）
# 例如：['\n', ' ', '!', ... 'a', 'b', ...]
chars = sorted(list(set(text)))

# vocab_size：字符表大小 V
vocab_size = len(chars)

# stoi: char -> id（string to int）
# itos: id -> char（int to string）
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    """
    把字符串 s 编码为 token id 列表（字符级）
    输入:  s: "hello"
    输出: [id('h'), id('e'), id('l'), id('l'), id('o')]
    """
    return [stoi[c] for c in s]

def decode(ids):
    """
    把 token id 列表解码回字符串
    输入:  [12, 5, 9]
    输出: "abc"（取决于词表映射）
    """
    return "".join([itos[i] for i in ids])

# 把整篇文本变成一个很长的 token 序列 data: [N]
# dtype=torch.long 是因为 embedding/cross_entropy 需要整数类别 id
data = torch.tensor(encode(text), dtype=torch.long)

# 划分 train/val（按时间顺序切分，避免泄漏）
n = int(cfg.train_split * len(data))
train_data = data[:n]
val_data = data[n:]


# ----------------------------
# 3) 构造 batch：随机切片做 next-token 监督
# ----------------------------
def get_batch(split: str):
    """
    构造一个 batch 的训练样本 (x, y)

    训练数据是一个长序列 train_data: [N]
    我们随机挑 batch_size 个起点 i，然后取：
        x = src[i : i+T]
        y = src[i+1 : i+T+1]
    也就是 y 是 x 的“整体右移一位”，用于 next-token prediction。

    返回:
        x: [B, T]  输入 token 序列
        y: [B, T]  目标 token 序列（下一个字符）
    """
    src = train_data if split == "train" else val_data

    # 随机选 batch_size 个起点
    # 起点范围必须保证切片不会越界：i + block_size + 1 <= len(src)
    ix = torch.randint(0, len(src) - cfg.block_size - 1, (cfg.batch_size,))

    # x: [B, T]   y: [B, T]
    # x[b] 是 src 从 ix[b] 开始的长度为 T 的片段
    # y[b] 是对应片段向右平移一位（真实下一个字符）
    x = torch.stack([src[i:i + cfg.block_size] for i in ix])
    y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])

    # 放到 GPU/CPU
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def estimate_loss(model: nn.Module):
    """
    在 train 与 val 上估计 loss，用于监控训练是否过拟合/是否收敛。

    @torch.no_grad() 表示不记录梯度（节省显存/加速）
    model.eval()      表示关闭 dropout（评估更稳定）
    """
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters, device=cfg.device)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)   # forward 返回 logits 和 loss
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()  # 切回训练模式（开启 dropout 等）
    return out


# ----------------------------
# 4) Transformer 组件：注意力 / FFN / Block
# ----------------------------
class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）

    输入:
        x: [B, T, C]  B=batch, T=序列长度, C=embedding维度
    输出:
        y: [B, T, C]  经过 attention 融合上下文后的表示

    关键点：
    - "self-attention": Q/K/V 都来自同一个输入 x
    - "causal": 位置 t 只能看 [0..t]，不能看未来 [t+1..T-1]，否则会信息泄露
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        # 要求 C 能被头数整除，保证每个 head 的维度一致
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head  # 每个头的维度 hd

        # qkv: 一次线性变换输出 3C，用于拆成 Q/K/V
        # 输入:  [B, T, C]
        # 输出:  [B, T, 3C]
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # 输出投影：把多头合并后的 [B, T, C] 再线性映射回 C
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        # dropout：attention 权重与残差输出的 dropout
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask：下三角矩阵为 1，上三角为 0
        # 用于屏蔽“未来 token”
        mask = torch.tril(torch.ones(block_size, block_size))

        # register_buffer：这是一个张量，但不是可训练参数（不会被 optimizer 更新）
        # 好处：会随着 model.to(device) 自动搬到 GPU/CPU
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # 生成 qkv: [B, T, 3C]，然后切开成 q,k,v 各 [B, T, C]
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # 多头拆分：
        # [B, T, C] -> [B, T, nh, hd] -> transpose -> [B, nh, T, hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力分数：
        # q @ k^T:
        #   q: [B, nh, T, hd]
        #   k^T: [B, nh, hd, T]
        # => att: [B, nh, T, T]
        # / sqrt(hd) 是为了数值稳定（防止 dot product 随维度变大而爆）
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask：对未来位置置 -inf，使 softmax 后概率为 0
        # self.mask[:T, :T] 是为了兼容 T < block_size 的情况
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # softmax 变成注意力权重（每行和为 1）
        att = F.softmax(att, dim=-1)

        # 对权重做 dropout（训练时）
        att = self.attn_drop(att)

        # attention 加权求和：
        # att: [B, nh, T, T]
        # v:   [B, nh, T, hd]
        # => y: [B, nh, T, hd]
        y = att @ v

        # 合并多头：
        # [B, nh, T, hd] -> transpose -> [B, T, nh, hd] -> view -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + dropout
        y = self.resid_drop(self.proj(y))
        return y


class FeedForward(nn.Module):
    """
    前馈网络（FFN / MLP）

    输入:
        x: [B, T, C]
    输出:
        y: [B, T, C]

    作用：
    - attention 负责“信息在 token 之间混合”
    - FFN 负责“每个 token 位置的非线性变换与表达能力提升”
    - 典型结构：C -> 4C -> C + 激活函数（GELU）
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩维提升容量
            nn.GELU(),                      # 非线性
            nn.Linear(4 * n_embd, n_embd),  # 降维回 C
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer Block（预归一化 Pre-LN）

    结构：
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))

    输入/输出：
        [B, T, C] -> [B, T, C]

    作用：
        - 残差连接保证梯度更顺畅，训练更稳定
        - LayerNorm 稳定各层激活分布
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-LN：先 LN 再子层，再做残差
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ----------------------------
# 5) 最小 GPT：Embedding + N个Block + LM Head
# ----------------------------
class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT-like 语言模型（最小实现）

    输入（训练/推理）：
        idx: [B, T] token id 序列
    输出（训练）：
        logits: [B, T, V]
        loss: 标量（如果给 targets）
    输出（推理生成）：
        通过 generate() 不断采样 next token

    关键组件：
    - token embedding：把 token id -> 向量
    - position embedding：告诉模型序列位置信息
    - N 层 Transformer blocks
    - lm_head：把隐藏向量映射到 vocab logits
    """
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # token embedding: [V] -> [C]
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)

        # position embedding: [T] -> [C]
        # 注意：这里假设最大位置就是 block_size-1
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)

        # 堆叠 n_layer 个 Transformer block
        self.blocks = nn.Sequential(*[
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])

        # 最后再做一次 LayerNorm
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 语言模型头：把隐藏状态映射到 vocab logits
        # 输出: [B, T, V]
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        # 初始化参数（与很多 GPT 实现类似的正态初始化）
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 线性层、Embedding 的权重初始化
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播

        输入:
            idx: [B, T] token ids
            targets: [B, T]（可选）训练时的真实下一个 token ids
        输出:
            logits: [B, T, V]
            loss:   标量（如果 targets 不为 None）
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, "序列长度不能超过 block_size"

        # 位置索引 pos: [T]
        # 例如 T=4 -> [0,1,2,3]
        pos = torch.arange(0, T, device=idx.device)

        # token embedding: [B, T] -> [B, T, C]
        # pos embedding:   [T]    -> [T, C] 会广播到 [B, T, C]
        x = self.token_emb(idx) + self.pos_emb(pos)

        # dropout（训练时生效）
        x = self.drop(x)

        # Transformer blocks: [B, T, C] -> [B, T, C]
        x = self.blocks(x)

        # final layer norm
        x = self.ln_f(x)

        # lm head: [B, T, C] -> [B, T, V]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 交叉熵需要：
            #   input:  [N, V]
            #   target: [N]
            # 这里把 [B, T, V] 展平成 [B*T, V]，
            # targets: [B, T] -> [B*T]
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size),
                targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        自回归生成（采样）

        输入:
            idx: [B, T] 初始上下文 token ids
        输出:
            idx: [B, T + max_new_tokens] 追加生成后的 token ids

        生成逻辑：
            重复 max_new_tokens 次：
            1) 把当前序列截断到最后 block_size 个 token（模型上下文上限）
            2) 前向得到 logits
            3) 取最后一个位置的 logits -> softmax -> 得到下一个 token 概率
            4) multinomial 按概率采样一个 token
            5) 拼到序列末尾
        """
        for _ in range(max_new_tokens):
            # 截断上下文，避免超过模型最大上下文长度
            idx_cond = idx[:, -self.cfg.block_size:]

            # logits: [B, Tcond, V]
            logits, _ = self(idx_cond)

            # 取最后一个时间步的 logits: [B, V]
            logits_last = logits[:, -1, :]

            # softmax 转概率分布
            probs = F.softmax(logits_last, dim=-1)

            # 从概率分布采样一个 token id: [B, 1]
            next_id = torch.multinomial(probs, num_samples=1)

            # 拼接到原序列末尾: [B, T+1]
            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ----------------------------
# 6) 训练主程序
# ----------------------------
# 构建模型并放到 device
model = GPTLanguageModel(vocab_size, cfg).to(cfg.device)

# AdamW 优化器：Transformer 训练常用
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay
)

t0 = time.time()
for it in range(1, cfg.max_iters + 1):

    # 每隔 eval_interval 步评估一次，或第一步也评估
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
        t0 = time.time()

    # 取一个训练 batch
    xb, yb = get_batch("train")          # xb/yb: [B, T]

    # 前向：得到 logits 与 loss
    logits, loss = model(xb, yb)         # logits: [B, T, V], loss: 标量

    # 清空梯度（set_to_none=True 省显存/更快）
    optimizer.zero_grad(set_to_none=True)

    # 反向传播：计算所有参数的梯度
    loss.backward()

    # 梯度裁剪：防止梯度爆炸（把整体梯度范数限制在 grad_clip 内）
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    # 参数更新
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
# 7) 训练结束：做一次生成看看模型学到什么
# ----------------------------
# context: [1, 1] 从一个起始 token 开始生成
# 这里用 token=0 作为起点（对应词表中的第一个字符）
context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

# generate 输出: [1, 1 + gen_len]，取第 0 条并转为 Python list
out = model.generate(context, cfg.gen_len)[0].tolist()

print("\n=== sample ===")
print(decode(out))  # 把 token ids 解码回字符串输出
sample_text = decode(out)
wandb.log({"sample/text": sample_text})
wandb.finish()
