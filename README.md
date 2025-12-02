# HOPE Model: Hierarchical Optimizer with Persistent Enhancement

基于 NL (Nested Learning) 论文实现的 HOPE 模型，整合了自修改序列模型和连续内存系统。

## 项目概述

HOPE (Hierarchical Optimizer with Persistent Enhancement) 是一个基于嵌套学习范式的神经网络模型，实现了以下核心特性：

1. **嵌套学习架构**：多层级优化问题，每层有自己的上下文流
2. **连续内存系统**：多层次内存银行（Ultra-short, Short, Mid, Long, Episodic）
3. **自修改序列模型**：学习如何修改自己的更新算法
4. **深度优化器**：多时间尺度的梯度压缩和参数同步

## 项目结构

```
hope_model/
├── Cargo.toml              # 项目配置
├── README.md               # 本文档
├── src/
│   ├── main.rs            # 训练入口和 CLI
│   ├── config.rs          # 配置结构定义
│   ├── model/
│   │   ├── mod.rs         # 模块导出
│   │   ├── hope.rs        # HOPE 主模型
│   │   ├── self_modify.rs # 自修改序列模型
│   │   ├── continuum_mem.rs # 连续内存系统
│   │   └── optimizer.rs   # Deep Optimizer
│   └── training/
│       ├── mod.rs
│       └── trainer.rs     # 训练循环
└── examples/
    ├── config_hope.json   # HOPE 配置示例
    └── train_hope.sh      # 训练脚本
```

## 快速开始

### 1. 安装依赖

确保已安装 Rust 工具链（1.81+）：

```bash
rustup component add rustfmt clippy
```

### 2. 构建项目

```bash
cargo build --release
```

### 3. 运行训练

使用示例配置文件：

```bash
cargo run --release --bin hope-train -- train --config examples/config_hope.json
```

或使用提供的脚本：

```bash
bash examples/train_hope.sh
```

## 配置说明

配置文件采用 JSON 格式，主要包含两部分：

### 模型配置 (`model`)

- `hidden_size`: 隐藏层维度（默认：384）
- `vocab_size`: 词汇表大小（默认：512）
- `seq_len`: 序列长度（默认：256）
- `num_levels`: 嵌套层级数（默认：3）
- `level_timescales`: 每层的更新频率，例如 `[1, 4, 16]`

#### 连续内存系统 (`continuum_mem`)

- `enabled`: 是否启用（默认：true）
- `ultra_short_span`: Ultra-short 内存跨度（默认：2）
- `short_span`: Short 内存跨度（默认：8）
- `mid_span`: Mid 内存跨度（默认：32）
- `long_span`: Long 内存跨度（默认：128）
- `episodic_span`: Episodic 内存跨度（默认：512）

#### 自修改模块 (`self_modify`)

- `enabled`: 是否启用（默认：true）
- `meta_lr`: 元学习率（默认：1e-5）
- `update_frequency`: 更新频率（默认：8）
- `weight_mod_dim`: 权重修改网络维度（默认：128）

#### 深度优化器 (`deep_optimizer`)

- `enabled`: 是否启用（默认：true）
- `fast_lr_scale`: 快速学习率缩放（默认：1.0）
- `slow_lr_scale`: 慢速学习率缩放（默认：0.1）
- `fast_ema`: 快速 EMA 系数（默认：0.9）
- `slow_ema`: 慢速 EMA 系数（默认：0.99）
- `sync_interval`: 同步间隔（默认：64）
- `gradient_compression_dim`: 梯度压缩维度（默认：256）

### 训练配置 (`training`)

- `batch_size`: 批次大小（默认：4）
- `learning_rate`: 学习率（默认：1e-4）
- `num_steps`: 训练步数（默认：1000）
- `log_every`: 日志输出间隔（默认：10）
- `use_random_data`: 是否使用随机数据（默认：true）

## 核心概念

### 嵌套学习 (Nested Learning)

HOPE 模型采用嵌套学习架构，将模型表示为多层级的优化问题。每一层都有自己的上下文流和更新频率，实现了多时间尺度的学习。

### 连续内存系统

传统的长/短期内存被推广为连续的多层次内存银行：

- **Ultra-short**: 1-4 步的记忆，用于即时上下文
- **Short**: 4-16 步的记忆，用于短期模式
- **Mid**: 16-64 步的记忆，用于中期依赖
- **Long**: 64-256 步的记忆，用于长期结构
- **Episodic**: >256 步的记忆，用于情节记忆

### 自修改模型

模型学习如何修改自己的更新规则，通过元学习网络生成权重修改，实现自适应优化。

### 深度优化器

实现梯度压缩和多时间尺度同步，通过快慢两种学习率实现更稳定的优化过程。

## 模型规模

默认配置下的模型大小约为 **1-10M 参数**，适合：
- 概念验证和实验
- 小规模任务
- 快速迭代开发

## 技术栈

- **框架**: Burn 0.19
- **后端**: `Autodiff<NdArray<f32>>` (CPU) 或 `Autodiff<Wgpu>` (GPU)
- **序列建模**: Transformer 编码器
- **优化器**: Adam + Deep Optimizer 扩展

## 开发计划

- [ ] 完整的数据加载器
- [ ] 模型检查点保存/加载
- [ ] 评估指标和可视化
- [ ] GPU 后端支持
- [ ] 更多任务适配（语言建模、持续学习等）

## 参考文献

基于论文：**Nested Learning: The Illusion of Deep Learning Architectures** (NL.pdf)

## License

Apache-2.0

