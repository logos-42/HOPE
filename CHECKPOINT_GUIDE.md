# HOPE 模型检查点使用指南

本指南介绍如何使用 HOPE 模型的检查点（Checkpoint）功能进行断点续训。

## 功能概述

检查点系统允许你：
- 在训练过程中定期保存模型状态
- 从中断的训练中恢复
- 保存最佳模型用于后续使用
- 进行实验对比和模型版本管理

## 检查点内容

每个检查点包含：
1. **模型权重**：所有可训练参数
2. **训练步数**：当前训练进度
3. **配置信息**：模型和训练配置
4. **时间戳**：创建时间

## 配置检查点

### 基础配置

在训练配置中添加检查点相关参数：

```json
{
  "training": {
    "checkpoint_dir": "./checkpoints",
    "save_every": 100,
    "resume_from": null
  }
}
```

**参数说明：**
- `checkpoint_dir`: 检查点保存目录（默认：`./checkpoints`）
- `save_every`: 每N步保存一次检查点（默认：100）
- `resume_from`: 从指定检查点恢复训练（可选）

## 使用示例

### 1. 开始新训练

配置文件 `config.json`：

```json
{
  "model": { ... },
  "training": {
    "batch_size": 4,
    "num_steps": 1000,
    "checkpoint_dir": "./checkpoints",
    "save_every": 100,
    "resume_from": null
  }
}
```

运行训练：

```bash
cargo run --release --bin hope-train -- train --config config.json
```

训练过程中会自动保存检查点：
- 每 100 步保存一次
- 训练结束时保存最终检查点

### 2. 从检查点恢复训练

假设训练在步骤 500 中断，检查点目录中有：

```
checkpoints/
  checkpoint_step_100_ts_1234567890.json
  checkpoint_step_100_ts_1234567890_model.mpk
  checkpoint_step_200_ts_1234567891.json
  checkpoint_step_200_ts_1234567891_model.mpk
  checkpoint_step_300_ts_1234567892.json
  checkpoint_step_300_ts_1234567892_model.mpk
  checkpoint_step_400_ts_1234567893.json
  checkpoint_step_400_ts_1234567893_model.mpk
  checkpoint_step_500_ts_1234567894.json
  checkpoint_step_500_ts_1234567894_model.mpk
```

修改配置文件：

```json
{
  "training": {
    "resume_from": "./checkpoints/checkpoint_step_500_ts_1234567894.json"
  }
}
```

继续训练：

```bash
cargo run --release --bin hope-train -- train --config config.json
```

训练将从步骤 500 继续。

### 3. 查看可用检查点

程序启动时会自动列出可用的检查点：

```
INFO  Found 5 existing checkpoint(s) in "./checkpoints"
INFO  Latest checkpoint at step: 500
INFO  Resuming training from checkpoint: "./checkpoints/checkpoint_step_500_ts_1234567894.json"
INFO  Resumed from step 500
```

## 检查点文件结构

### 元数据文件 (.json)

```json
{
  "step": 500,
  "timestamp": 1234567894,
  "model_file": "checkpoint_step_500_ts_1234567894_model",
  "config": {
    "model": { ... },
    "training": { ... }
  }
}
```

### 模型文件 (.mpk)

二进制格式，包含所有模型权重。

## 最佳实践

### 1. 保存频率

根据训练时长和资源选择合适的保存频率：

- **快速实验**：每 10-50 步
- **正常训练**：每 100-500 步
- **长期训练**：每 1000-5000 步

```json
{
  "training": {
    "save_every": 100  // 根据需要调整
  }
}
```

### 2. 磁盘空间管理

检查点会占用磁盘空间，定期清理旧检查点：

```bash
# 保留最近的 5 个检查点
cd checkpoints
ls -t checkpoint_*.json | tail -n +6 | xargs rm -f
ls -t checkpoint_*.mpk | tail -n +6 | xargs rm -f
```

### 3. 备份重要检查点

将关键检查点备份到安全位置：

```bash
# 备份最佳模型
cp checkpoints/checkpoint_step_5000_*.* backups/best_model/
```

### 4. 版本管理

为不同实验创建独立的检查点目录：

```json
{
  "training": {
    "checkpoint_dir": "./checkpoints/experiment_v1"
  }
}
```

## 常见场景

### 场景1：训练中断恢复

**问题**：训练因断电/错误中断

**解决**：
1. 查看最新检查点
2. 更新 `resume_from` 配置
3. 重新运行训练

### 场景2：调整训练参数

**问题**：想要改变学习率继续训练

**解决**：
1. 从检查点恢复
2. 修改配置中的 `learning_rate`
3. 继续训练

**注意**：某些参数（如模型结构）不能改变。

### 场景3：多阶段训练

**问题**：先训练 1000 步，然后再训练 2000 步

**解决**：

第一阶段：
```json
{
  "training": {
    "num_steps": 1000,
    "resume_from": null
  }
}
```

第二阶段：
```json
{
  "training": {
    "num_steps": 2000,
    "resume_from": "./checkpoints/checkpoint_step_1000_*.json"
  }
}
```

### 场景4：模型评估

**问题**：在不同检查点评估模型性能

**解决**：
1. 保存多个检查点
2. 对每个检查点运行评估
3. 选择最佳检查点

## 故障排查

### 问题1：检查点加载失败

**症状**：
```
Error: Failed to load checkpoint
```

**可能原因**：
- 文件路径错误
- 文件损坏
- 模型配置不匹配

**解决方案**：
1. 检查文件路径是否正确
2. 验证文件完整性
3. 确保配置与检查点一致

### 问题2：配置不匹配

**症状**：
```
Error: Checkpoint model config doesn't match current config
```

**原因**：检查点的模型配置与当前配置不同

**解决方案**：
- 使用检查点中保存的配置
- 或从头开始训练新模型

### 问题3：磁盘空间不足

**症状**：保存检查点失败

**解决方案**：
1. 清理旧检查点
2. 增加保存间隔
3. 使用更大的磁盘

### 问题4：检查点文件损坏

**症状**：加载时出现解析错误

**解决方案**：
- 使用更早的检查点
- 检查磁盘健康状况
- 确保训练时有足够的磁盘空间

## 高级用法

### 自动选择最新检查点

如果不想手动指定检查点，可以使用脚本自动选择：

```bash
# 找到最新的检查点
LATEST=$(ls -t checkpoints/checkpoint_*.json | head -n 1)

# 更新配置（使用 jq 工具）
jq ".training.resume_from = \"$LATEST\"" config.json > config_resume.json

# 运行训练
cargo run --release --bin hope-train -- train --config config_resume.json
```

### 检查点转换

将检查点转换为其他格式（未来功能）：

```bash
# 导出为 ONNX（计划中）
cargo run --bin export-model -- \
  --checkpoint ./checkpoints/checkpoint_step_1000.json \
  --format onnx \
  --output model.onnx
```

## 检查点与版本控制

### 建议做法

1. **不要**将检查点文件提交到 Git
2. 在 `.gitignore` 中添加：
   ```
   checkpoints/
   *.mpk
   ```
3. 使用专门的模型存储服务（如 DVC、Git LFS）

### 共享检查点

如需共享检查点：
1. 压缩检查点文件
2. 上传到云存储
3. 提供下载链接和使用说明

## 性能考虑

### 保存时间

- 小模型（< 10M 参数）：< 1 秒
- 中等模型（10-100M 参数）：1-5 秒
- 大模型（> 100M 参数）：5-30 秒

### 文件大小

估算公式：`文件大小 ≈ 参数数量 × 4 字节`

示例：
- 10M 参数 ≈ 40 MB
- 100M 参数 ≈ 400 MB

## 总结

检查点系统是训练长期模型的关键功能：

✅ **优点**：
- 防止训练中断导致的损失
- 支持实验迭代和参数调优
- 便于模型版本管理

⚠️ **注意**：
- 定期清理旧检查点
- 备份重要模型
- 确保充足的磁盘空间

📚 **相关文档**：
- [主要训练指南](README.md)
- [书籍训练指南](BOOK_TRAINING_GUIDE.md)

