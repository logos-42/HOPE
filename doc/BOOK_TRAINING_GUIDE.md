# HOPE 模型书籍训练指南

本指南介绍如何使用 PDF 和 EPUB 书籍训练 HOPE 模型。

## 目录

1. [准备书籍数据](#准备书籍数据)
2. [预处理步骤](#预处理步骤)
3. [训练配置](#训练配置)
4. [训练命令](#训练命令)
5. [OCR 配置](#ocr-配置)
6. [故障排查](#故障排查)

## 准备书籍数据

### 支持的格式

- **PDF 文件** (.pdf)
  - 文字型 PDF：可直接提取文本
  - 扫描版 PDF：需要 OCR 识别

- **EPUB 文件** (.epub)
  - 标准电子书格式
  - 自动提取章节结构

### 组织数据

将所有书籍文件放在一个目录中：

```
data/
  books/
    book1.pdf
    book2.epub
    book3.pdf
    ...
```

## 预处理步骤

### 方法1：离线预处理（推荐）

适用于大量书籍，一次性转换为文本格式：

```bash
# 编译预处理工具
cargo build --release --bin preprocess-books

# 运行预处理
./target/release/preprocess-books \
  --input ./data/books \
  --output ./data/preprocessed \
  --preserve-structure true \
  --enable-ocr false
```

**参数说明：**
- `--input`: 输入目录（包含 PDF/EPUB 文件）
- `--output`: 输出目录（生成的文本文件）
- `--preserve-structure`: 是否保留章节段落结构标记
- `--enable-ocr`: 是否启用 OCR（处理扫描版 PDF）

**输出文件：**
- `corpus.jsonl`: 所有文档的 JSON Lines 格式
- `vocab.json`: 字符级词汇表
- `metadata.json`: 语料库统计信息
- `*.txt`: 每本书的文本文件

### 方法2：在线加载

训练时实时解析书籍（适合小规模数据）：

直接在配置文件中指定书籍目录，训练时自动解析。

## 训练配置

### 基础配置

创建 `config_books.json`：

```json
{
  "model": {
    "hidden_size": 384,
    "vocab_size": 512,
    "seq_len": 256,
    "num_heads": 8,
    "num_layers": 4,
    "ff_multiplier": 4.0,
    "dropout": 0.1,
    "num_levels": 3,
    "level_timescales": [1, 4, 16],
    "continuum_mem": {
      "enabled": true,
      "ultra_short_span": 2,
      "short_span": 8,
      "mid_span": 32,
      "long_span": 128,
      "episodic_span": 512
    },
    "self_modify": {
      "enabled": true,
      "meta_lr": 1e-5,
      "update_frequency": 8,
      "weight_mod_dim": 128
    },
    "deep_optimizer": {
      "enabled": true,
      "fast_lr_scale": 1.0,
      "slow_lr_scale": 0.1,
      "fast_ema": 0.9,
      "slow_ema": 0.99,
      "sync_interval": 64,
      "gradient_compression_dim": 256
    }
  },
  "training": {
    "batch_size": 4,
    "learning_rate": 1e-4,
    "num_steps": 10000,
    "log_every": 100,
    "use_random_data": false,
    "checkpoint_dir": "./checkpoints",
    "save_every": 1000,
    "resume_from": null
  },
  "data": {
    "data_type": "books",
    "data_path": "./data/books",
    "tokenizer_path": "./data/preprocessed/vocab.json"
  }
}
```

### 配置说明

**模型配置：**
- `vocab_size`: 根据预处理生成的词汇表大小调整
- `seq_len`: 序列长度，建议 256-512
- `hidden_size`: 隐藏层维度，影响模型容量

**训练配置：**
- `use_random_data`: 设为 `false` 使用真实数据
- `checkpoint_dir`: 检查点保存目录
- `save_every`: 每N步保存一次检查点
- `resume_from`: 从检查点恢复训练（可选）

**数据配置：**
- `data_type`: `"books"` 表示使用书籍数据
- `data_path`: 书籍文件目录或预处理输出目录
- `tokenizer_path`: 词汇表文件路径

## 训练命令

### 从头开始训练

```bash
cargo run --release --bin hope-train -- train --config config_books.json
```

### 从检查点恢复训练

修改配置文件中的 `resume_from`：

```json
{
  "training": {
    ...
    "resume_from": "./checkpoints/checkpoint_step_1000_ts_1234567890.json"
  }
}
```

然后运行：

```bash
cargo run --release --bin hope-train -- train --config config_books.json
```

## OCR 配置

### 安装 Tesseract OCR

**Windows:**
1. 下载：https://github.com/UB-Mannheim/tesseract/wiki
2. 安装并添加到 PATH
3. 下载语言包（中文：chi_sim, 英文：eng）

**Linux:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

**Mac:**
```bash
brew install tesseract tesseract-lang
```

### 安装 Poppler（PDF 转图像）

**Windows:**
1. 下载：https://github.com/oschwartz10612/poppler-windows/releases/
2. 解压并添加 bin 目录到 PATH

**Linux:**
```bash
sudo apt-get install poppler-utils
```

**Mac:**
```bash
brew install poppler
```

### 验证安装

```bash
# 检查 Tesseract
tesseract --version

# 检查 Poppler
pdftoppm -v
```

### 使用 OCR

在预处理时启用 OCR：

```bash
./target/release/preprocess-books \
  --input ./data/books \
  --output ./data/preprocessed \
  --enable-ocr true
```

**注意：** OCR 处理速度较慢，建议先对小批量数据测试。

## 故障排查

### 问题1：PDF 无法提取文本

**症状：** "PDF has no extractable text"

**解决方案：**
1. 检查 PDF 是否为扫描版
2. 启用 OCR：`--enable-ocr true`
3. 确保 Tesseract 和 Poppler 已安装

### 问题2：内存不足

**症状：** 训练时内存溢出

**解决方案：**
1. 减少 `batch_size`（如 4 → 2）
2. 减少 `seq_len`（如 512 → 256）
3. 减少 `hidden_size`（如 512 → 384）
4. 使用预处理后的数据而非在线加载

### 问题3：训练速度慢

**症状：** 每步训练时间过长

**解决方案：**
1. 使用 `--release` 模式编译
2. 减少模型复杂度（层数、维度）
3. 考虑使用 GPU 后端（需要配置 WGPU 或 TCH）
4. 使用预处理数据避免实时解析

### 问题4：词汇表大小不匹配

**症状：** "vocab size mismatch"

**解决方案：**
1. 确保配置中的 `vocab_size` 与 `vocab.json` 中的大小一致
2. 重新运行预处理生成新的词汇表
3. 或使用预处理生成的词汇表大小更新配置

### 问题5：检查点加载失败

**症状：** "Failed to load checkpoint"

**解决方案：**
1. 检查检查点文件路径是否正确
2. 确保模型配置与检查点一致
3. 检查检查点文件是否完整（未损坏）

## 最佳实践

### 数据准备

1. **清理数据**：移除损坏或无法读取的文件
2. **分类组织**：按主题或类型组织书籍
3. **测试小样本**：先用少量书籍测试流程

### 训练策略

1. **渐进式训练**：
   - 先用小模型快速验证
   - 逐步增加模型规模
   - 使用检查点继续训练

2. **监控指标**：
   - 观察 Loss 下降趋势
   - 定期保存检查点
   - 记录训练日志

3. **资源管理**：
   - 预估内存需求
   - 合理设置批次大小
   - 定期清理旧检查点

### 性能优化

1. **预处理优先**：大规模数据务必使用离线预处理
2. **批次调优**：找到内存和速度的最佳平衡点
3. **检查点策略**：根据训练时长设置保存频率

## 示例工作流

### 完整训练流程

```bash
# 1. 准备数据
mkdir -p data/books
# 将 PDF/EPUB 文件复制到 data/books/

# 2. 预处理
cargo build --release --bin preprocess-books
./target/release/preprocess-books \
  --input ./data/books \
  --output ./data/preprocessed \
  --preserve-structure true

# 3. 创建配置文件
# 编辑 config_books.json，设置正确的 vocab_size

# 4. 开始训练
cargo run --release --bin hope-train -- train --config config_books.json

# 5. 监控训练
# 查看日志输出，观察 Loss 变化

# 6. 从检查点恢复（如需要）
# 修改 config_books.json 中的 resume_from
# 重新运行训练命令
```

## 进阶话题

### 多语言支持

修改 OCR 语言参数（在 `src/utils/ocr.rs` 中）：

```rust
.arg("-l")
.arg("chi_sim+eng")  // 中文简体 + 英文
```

### 自定义结构标记

修改 `src/utils/text_processor.rs` 中的标记格式。

### GPU 加速

启用 WGPU 后端：

```bash
cargo build --release --features wgpu-backend
```

更新配置以使用 GPU。

## 参考资源

- [Tesseract OCR 文档](https://github.com/tesseract-ocr/tesseract)
- [Poppler 工具](https://poppler.freedesktop.org/)
- [HOPE 模型主文档](README.md)

## 获取帮助

如遇问题，请检查：
1. 日志输出中的错误信息
2. 本故障排查部分
3. 确保所有依赖已正确安装

