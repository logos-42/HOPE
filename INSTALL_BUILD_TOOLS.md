# 解决 Rust 编译问题：安装 Visual Studio Build Tools

## 问题描述

编译 Rust 项目时出现错误：
```
error: linker `link.exe` not found
note: the msvc targets depend on the msvc linker but `link.exe` was not found
```

这是因为 Windows 上的 Rust 默认使用 MSVC 工具链，需要 Visual Studio Build Tools。

## 解决方案

### 方案 1：安装 Visual Studio Build Tools（推荐）

1. **下载 Visual Studio Build Tools**
   - 访问：https://visualstudio.microsoft.com/downloads/
   - 下载 "Build Tools for Visual Studio 2022"

2. **安装时选择组件**
   - 运行安装程序
   - 选择 "C++ build tools" 工作负载
   - 确保包含以下组件：
     - MSVC v143 - VS 2022 C++ x64/x86 build tools
     - Windows 10/11 SDK
     - C++ CMake tools for Windows

3. **重启终端**
   - 安装完成后，关闭并重新打开 PowerShell/CMD
   - 运行 `cargo build` 应该可以正常编译

### 方案 2：切换到 GNU 工具链（替代方案）

如果不想安装 Visual Studio，可以切换到 GNU 工具链：

```powershell
# 安装 GNU 工具链
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu

# 安装 mingw-w64
# 使用 Chocolatey（如果已安装）：
choco install mingw

# 或手动下载安装：
# https://www.mingw-w64.org/downloads/
```

然后设置环境变量：
```powershell
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER = "x86_64-w64-mingw32-gcc"
```

### 方案 3：使用已编译的版本

如果 `target\debug\hope-train.exe` 已存在且可以运行，可以直接使用：

```powershell
.\target\debug\hope-train.exe train --config examples\config_hope.json
```

## 验证安装

安装完成后，验证链接器是否可用：

```powershell
# 检查 MSVC 链接器
where link.exe

# 或检查 Rust 工具链
rustup show
```

## 快速安装命令（使用 Chocolatey）

如果已安装 Chocolatey，可以快速安装：

```powershell
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## 注意事项

- Visual Studio Build Tools 大约需要 3-6 GB 磁盘空间
- 安装后需要重启终端才能生效
- 如果使用 WSL，可以在 Linux 环境中编译，避免 Windows 工具链问题

