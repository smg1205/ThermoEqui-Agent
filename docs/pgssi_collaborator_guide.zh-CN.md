# PGSSI 协作者测试交接文档（傻瓜版）

> 这份文档告诉你：拿到一个权重文件后，如何让 PGSSI 在你自己的电脑上跑起来并测试。
> 按顺序照做即可，不需要理解原理。

---

## 1. PGSSI 是什么（30 秒理解）

PGSSI 是课题组训练的一个**机器学习模型**，专门做一件事：

> **给定"溶质 + 溶剂 + 温度"，预测无限稀释活度系数 γ∞**

（γ∞ 是衡量"某物质在另一物质中极度稀释时的行为"的物理量，是热力学的重要性质。）

- 输入：两个分子的 SMILES 结构式 + 温度
- 输出：γ∞（以及它的对数 ln γ∞）

它**不是**泡点/露点/闪蒸求解器，而是"性质预测器"。

---

## 2. 你需要准备的 3 样东西

| # | 东西 | 说明 |
|---|---|---|
| 1 | **项目代码** | `ThermoEqui-Agent` 仓库，切到 `feat/pgssi-gamma-curve` 分支 |
| 2 | **权重文件** | `all_merged_train_PGSSI_best.pth`（约 85 MB，别人给你的）|
| 3 | **PGSSI 源码** | 课题组 PGSSI 仓库（含 `PGSSI_3D_architecture.py` 的目录）|

> ⚠️ **权重是私有的，不要提交到 git、不要外传、不要发到公开仓库。** 它放在你本地任意位置即可。

---

## 3. 环境准备（一次性）

### 3.1 装依赖

用 Python 3.11/3.12 环境，安装：

```bash
pip install "torch" torch-geometric rdkit pandas scikit-learn tqdm
```

> 如果 `import torch_scatter` 报错，运行：
> ```bash
> pip install torch-scatter
> ```
> 装不上也不影响——项目内置了兼容垫片，会自动处理。

### 3.2 建 `.env` 文件（关键！）

在项目根目录创建 `.env` 文件（如果已有就编辑），填 4 行：

```
PGSSI_CHECKPOINT=D:\你的路径\all_merged_train_PGSSI_best.pth
PGSSI_SRC=D:\你的路径\PGSSI\src\models\PGSSI
PGSSI_HIDDEN_DIM=512
PGSSI_ENABLE_CROSS_INTERACTION=1
```

**每条的意思**：
- `PGSSI_CHECKPOINT`：权重文件的**完整路径**（改成你自己的）
- `PGSSI_SRC`：PGSSI 源码里 `src/models/PGSSI` 这个目录的完整路径（含 `PGSSI_3D_architecture.py` 的那个文件夹）
- 后两行是训练超参数，**照抄 512 和 1 即可**

> ⚠️ `.env` 已被 `.gitignore` 排除，不会被提交。但**不要**把真实路径发到群里。

---

## 4. 启动后端

```bash
cd ThermoEqui-Agent
python -m uvicorn apps.api.main:app --port 8000
```

> ⚠️ **重要**：如果你电脑上有多个 Python，务必用装了 `torch_geometric`/`rdkit` 的那个环境（比如 conda 环境 `D:\Anaconda\envs\thermoequi-dev\python.exe`）。用错环境会报"PGSSI requires optional dependencies"。

看到 `Application startup complete` 就是成功了。

---

## 5. 测试（复制粘贴即可）

### 5.1 直接测 API（最快验证）

新开一个终端：

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())"
```

返回 `{"status":"ok",...}` = 后端正常。

### 5.2 在聊天框测试（推荐）

打开前端 `http://localhost:3000`（或直接用 API 工具），输入下面的问题：

| 问题 | 预期结果 |
|---|---|
| **计算乙醇在水中298K的无限稀释活度系数** | `γ∞(Ethanol→Water) ≈ 2.70`，`γ∞(Water→Ethanol) ≈ 1.96` |
| **计算乙醇在水中280到360K的无限稀释活度系数曲线** | 一条 γ∞-T 曲线（2 个方向），温度 280→360K |
| **预测乙烷在丙烷中的γ∞ 310K** | 一个数值（量级 ~1.x）|
| **计算苯-甲苯在101.325 kPa下的T-x-y曲线** | 正常 T-x-y 相图（**不是 PGSSI**，是 Ideal/Raoult）|

**判断标准**：
- γ∞ 数值应该在 **0.5 ~ 10 之间**（合理物理范围）
- 曲线应随温度**缓慢变化**（不是乱跳）
- 如果返回 "PGSSI requires a trained checkpoint" → 权重路径没配好（检查 `.env`）
- 如果返回 "PGSSI requires optional dependencies" → Python 环境用错了（见第 4 步警告）

---

## 6. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `PGSSI requires a trained checkpoint but none is configured` | `.env` 里权重路径不对或没生效 | 检查 `PGSSI_CHECKPOINT` 路径 + 重启后端 |
| `PGSSI requires optional dependencies that are not installed` | 用错了 Python 环境 | 用装了 torch_geometric/rdkit 的环境启动 |
| `PGSSI model architecture is unavailable` | `PGSSI_SRC` 路径不对 | 检查 `PGSSI_SRC` 指向含 `PGSSI_3D_architecture.py` 的目录 |
| 前端显示 "Failed to fetch" | 后端没启动 / 端口不对 | 确认后端起来了，前端访问 8000 |
| 数值是 -600 或 0.0000 | 权重与代码架构不匹配 | 换用本项目配套的权重（AutoDL 训练版）|

---

## 7. 一句话总结

> **配好 `.env`（权重路径 + 源码路径）→ 用对 Python 环境启动 → 聊天框问 γ∞ 就能用。**

遇到问题把**后端终端完整输出**发给负责人，附上你的操作系统和 Python 版本即可。
