# 权重更新编程

一个关于 **Qwen3.5-0.8B 能否给自己写出有用的 LoRA** 的小型实验。模型读取自己的失败记录，选择低秩编辑配方，程序根据已有权重和激活计算更新，全程只做前向计算。

**目前没有观察到自编辑提升数学成绩。** 同一批 16 道留出题上，关闭 thinking 时原模型答对 9 道，加载自编辑 LoRA 后答对 7 道。两道原先答对的题在编辑后都用完了生成预算。这个结果限于当前样本、模型、配方和预算。[逐题结果](results/math-direct-1024/self_lora.jsonl) · [配对报告](results/math-direct-1024/report.json)

## 数学实验

从 GSM8K 训练集抽取 4 道练习题，再从 GSM8K 测试集和 MATH-500 分别抽取 8 道测试题。题号、顺序与采样种子固定，每题采样一次，最多生成 1,024 token。开启 thinking 时，只判定结束思考后最后一个完整的盒装答案。各组共用相同的评分规则。[评测协议与题号](results/math-thinking-1024/protocol.json)

| 作答方式 | 条件 | GSM8K | MATH-500 | 触及生成上限 |
| --- | --- | ---: | ---: | ---: |
| 开启 thinking | 原模型、经验提示词、自选 LoRA、固定 LoRA，各组结果相同 | 0/8 | 0/8 | 各 16/16 |
| 关闭 thinking | 原模型 | 6/8 | 3/8 | 6/16 |
| 关闭 thinking | 加载同一份自选 LoRA | 5/8 | 2/8 | 7/16 |

四道练习题都在思考结束前触顶。模型选择第 15 层 MLP 的 32 个输入列和 0.5 的编辑强度，层号从 0 开始。程序对未完成的推理和完整参考解分别采集末尾激活，用它们的均值差构造方向，在选中的列上减去相应投影。固定配方使用第 19 层、64 列和强度 1。该设置同时受到作答长度和终止行为影响。[配方与选择记录](results/math-thinking-1024/design.json) · [数学适配器](results/math-thinking-1024/adapter) · [四组报告](results/math-thinking-1024/report.json)

设计受 *Sparks of Weight-Space Programming* 的实验 8 启发。原文由外部 Astra 编辑 Qwen，测试使用 32,768-token 预算和多次采样。本项目测试同一个小模型能否通过受限配方编辑自己，分数适合在本项目的配对条件内比较。[原文](https://rikiyat.github.io/blog/weight-space-programming/)

## 复现

需要 Python 3.11、兼容的 PyTorch 与本地 Qwen3.5-0.8B 检查点。实测环境为 Windows、RTX 4060 Laptop 8 GB、PyTorch 2.7.0+cu126 和 Transformers 5.3.0。以下为 PowerShell 命令。先创建环境，按机器安装相应的 PyTorch，然后安装其余依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/Qwen3.5-0.8B
```

将三个固定版本的 JSONL 保存到 `data/math/`。评测脚本会校验文件的 SHA-256 及记录数。[GSM8K 数据源](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data) · [MATH-500 数据源](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)

```powershell
New-Item -ItemType Directory -Force data/math | Out-Null
$gsm = 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data'
curl.exe -fL "$gsm/train.jsonl" -o data/math/gsm8k_train.jsonl
curl.exe -fL "$gsm/test.jsonl" -o data/math/gsm8k_test.jsonl
curl.exe -fL 'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl' -o data/math/math500.jsonl

python -m unittest -v test_math test_experiment
python run_math.py --model models/Qwen3.5-0.8B --device cuda --out outputs/math-repeat
python eval_math_adapter.py --source outputs/math-repeat --out outputs/math-repeat-direct --device cuda
```

也可以把最后一条命令的 `--source` 换成 `results/math-thinking-1024`，直接复测仓库里的适配器。每次运行使用新的输出目录。公开的 [results](results) 包含配方、适配器、汇总和逐题判断。复跑时还会在本地保存题面和完整生成文本。适配器对应的基座为 Qwen3.5-0.8B。[基座模型](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [上游许可证](results/Qwen-LICENSE.txt)

## 早期合成诊断实验

早期实验研究自编辑能否帮助模型选择诊断探针并修复另一个小策略。在 24 个留出案例中，四组都完整修复了 12 个案例，而且两个决策都始终选择 A。LoRA 改变了输出概率，但没有改善修复质量。[实验报告](results/v1/report.json)

```powershell
python run.py --model models/Qwen3.5-0.8B --device cuda --candidates 2 --seed 19 --out outputs/diagnostic-repeat
```

### 一条数据怎样产生

1. 生成一个有三个数值输入的小策略。前两项代表工具状态与重试上下文，第三项是无关变量。健康策略计算前两项之和，并在分数至少为 1.5 时选择前进。
2. 破坏其中一个系数。开发阶段把 +1 改为 -2。测试除了同类故障，还包含一个域把 +1 改为 +4。编辑器知道两个可能的故障和数值变化，具体坏在哪一列由探针辨别。
3. 展示一个两个故障都会产生的事故记录。编辑器在两个输入中选一个进行反事实回放，由损坏策略做一次真实的数值前向计算并返回分数。编辑器据此从两个 rank-1 修复选项中选一个，框架将它加到目标的独立副本上。
4. 比较修复后和健康策略在 72 个数值状态上的动作。记录原本错误的动作恢复多少，以及原本正确的动作中有多少发生回退。72 个状态由 36 对信号取值与 2 个无关变量取值组成。

总计 48 条交互案例，按工作流分为发现集 16、选型验证集 8、留出测试集 24。每个域交叉两个故障位置、两个探针选项顺序和两个修复选项顺序。六种不同的初始探针提示构成了 24 个测试案例。状态格、场景和故障均为程序生成，没有使用外部业务轨迹。

### 自编辑的范围

冻结 Qwen 并关闭 autograd。采集未编辑模型在发现集选择探针时两个 MLP down_proj 的末 token 输入和输出。Qwen 读取自己的诊断记录后用四次 A/B 决策选择层号、读方向算法、写方向算法和强度。实验提供 16 种可能配方。编译器使用发现集正确探针标签与激活，求出读向量。写向量来自 A/B 输出嵌入差或本层激活差。读写向量的外积形成实际的 rank-1 LoRA。

模型选择配方，程序计算浮点因子。发现集有标签，适配过程不用梯度或优化器。执行测试时不再提供发现记录。自编辑 LoRA 在选择探针和选择修复两个决策中均启用。模型只能选择两种预定义的修复。

### 四组配对对照

| 组 | 诊断历史如何进入测试 | 候选选择 |
| --- | --- | --- |
| 原模型 | 无 | 直接测 |
| 经验提示词 | Qwen 根据发现记录生成一段短建议，测试时加入系统提示 | 2 个候选在验证集选优 |
| 自编辑 LoRA | Qwen 生成受限配方，编译器写适配器，测试加载落盘因子 | 同样生成并验证 2 次，允许重复 |
| 固定 LoRA | 同一发现集、标签和编译器，固定最后一层、均值差、输出嵌入方向和 0.25 强度 | 预设的一个对照 |

两种可选型方案使用相同的候选数和验证案例。实际生成 token、前向数和时间分别记录。每种方法在留出测试中的每题均有两次语言模型决策、一次目标探针和一次修复提交。测试前已冻结选型，模型生成的任意文字不会作为可执行代码运行。

### 读实验结果

正式运行保存在 `results/v1/`。

在 RTX 4060 Laptop 上，四组在 24 个留出案例中都选对 12 次探针，完整修好 12 次。每个组的 48 个决策全部选择 A。将同一道题的有效探针从 A 换到 B 后，自编辑 LoRA 给 A 的二选一概率分别为 77.7% 和 79.8%，原模型两次都是 59.3%。动作未随证据改变。每组恢复 480/720 个受损状态，导致 96/1008 个原本正确的状态回退。这些是交互中的重复检查次数。

两个自编辑候选生成了相同配方。模型选择索引为 12 的 MLP down_proj、类均值差读向量、输出嵌入差写向量及 0.25 强度。适配器有 4608 个参数。实验共 161.38 秒，PyTorch 峰值已分配显存 1566.2 MiB。此次实验没有观察到诊断能力或下一次编辑质量提升。

- `results/v1/report.json` 含最终计数、相对原模型逐题得失、各阶段开销、峰值 CUDA 已分配显存和运行环境。计时中的前向数指语言模型调用，目标策略调用不包含在内。测试阶段耗时不含加载适配器。显存计数来自 PyTorch。
- `results/v1/discovery.json` 含原模型在发现集上的决策与提供给它的诊断记录。
- `results/v1/validation.json` 含生成的提示词、LoRA 配方、逐题表现与选型索引。验证并列时优先探针选对数、保留正常行为的案例数，仍并列则取较早候选。
- `results/v1/test.json` 含四组在完全相同案例上的动作、A/B 归一化概率、修复结果和正常状态回退次数。
- `results/v1/adapter/` 含 LoRA 因子、适配器配置和配方。项目自身的加载器会从磁盘读回因子再评测。

这是一步实验。任务分布小、测试提示重复，发现与测试共享诊断规则。一次测试结果不足以判断真实工具使用、强基线上的收益或多代递归改进。资源消耗记录支持后续比较，当前无常规微调的等质量成本实测。
