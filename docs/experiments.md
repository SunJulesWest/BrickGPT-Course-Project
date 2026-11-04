# 批量实验框架

`run_experiments.py` 在课程仓库根目录运行，复用官方 `BrickGPTConfig`、
`BrickGPT.__call__()`、`transformers.set_seed()` 和结构导出接口。
不修改官方生成、拒绝采样或回退逻辑。此轮不实现结构、CLIP 或多样性评分。

## 无依赖试运行

只需要 Python 3.10 或更新版本，无需安装项目、Torch、Gurobi 或模型：

```sh
python run_experiments.py --prompts examples/experiment_prompts.jsonl --temperatures 0.4 0.6 --seeds 1 42 --mock
```

两条提示词、两个初始温度、两个种子构成 8 次实验。
`--mock` 只生成带明确标识的合成 TXT、LDraw 和结构 JSON，用于检查调度及文件流程。
它不模拟真实模型质量、拒绝采样轨迹或物理检查，不可用于论文结果。

保留失败并继续执行的示例：

```sh
python run_experiments.py --prompts examples/experiment_prompts.jsonl --temperatures 0.4 0.6 --seeds 1 42 --mock --mock-fail-prompt-id chair-01
```

此命令记录 4 次模拟失败和 4 次成功，退出码为 1。
提示词文件为 UTF-8 JSONL，每行具有唯一、非空 `id` 和非空 `text`。
支持中文及 UTF-8 BOM。ID 仅用作元数据，不参与拼接文件路径。

## 后续真实执行

在已按官方说明准备好依赖、模型访问与检测环境的 Linux 服务器中：

```sh
python run_experiments.py --prompts examples/experiment_prompts.jsonl --temperatures 0.4 0.6 --seeds 1 42 --config experiment_config.json
```

`experiment_config.json` 是官方 `BrickGPTConfig` 的覆盖参数对象，例如：

```json
{
  "model_name_or_path": "AvaLovelace/BrickGPT",
  "device": "cuda",
  "max_brick_rejections": 500,
  "max_regenerations": 100,
  "use_gurobi": true,
  "use_logit_masking": true,
  "temperature_increase": 0.01,
  "max_temperature": 2.0
}
```

这个配置文件仅为使用示例，本轮不会安装依赖或执行上述真实命令。
初始温度统一由 `--temperatures` 提供，配置中的 `temperature` 会被拒绝，避免歧义。
模型默认按官方规则加载，真实模式可能下载权重，必须在准备妥当后显式执行。
每个批次只加载一次模型，每次生成前设置初始温度和随机种子。
保持其他参数一致，温度变化为生成策略补充实验，不称作原论文既定实验。

默认只导出结构文件。显式传入 `--render` 才加载官方渲染模块生成 PNG，
该步骤需要 Blender、ImportLDraw 子模块、LDraw 零件库及官方背景资源。
mock CLI 不支持渲染。没有 Gurobi 时，可按官方设置 `use_gurobi: false`，
但其检查仅为连通性，不等同于物理稳定性。

## 输出与合并

默认输出 `experiments/runs/<batch_id>/`，已加入 `.gitignore`。
`--output-dir` 可以改为任何本地或服务器目录。每批使用 UUID，禁止覆盖同名批次目录。

- `manifest.json`：完整计划、提示词、种子、请求配置、代码提交号及工作区是否有改动。
- `records.jsonl`：每个已尝试实验先追加 `running` 事件，再追加终态事件。
- `<run_id>/structure.txt`、`structure.ldr`、`bricks.json`：依次导出的结果。
- `<run_id>/structure.png`：仅成功完成可选渲染时存在。

`run_id` 是与队友评价表合并的主键；同一批次内提示词由 `prompt_id` 标识。
每个事件含 `schema_version`、`batch_id`、`run_id`、提示词全文、种子、
`initial_temperature`、`requested_parameters`、`effective_config`、
`sampling_strategy`、`model_revision`、`source`、时间、状态、异常和结果路径。
`artifacts` 路径相对于该批次目录，采用 `/`，便于整体迁移到 Linux。
尚未成功导出的结果路径为 `null`；早先导出的文件不会因为后续失败而删除。

后续评价只取每个 `run_id` 的最后一条事件：

```python
import json

latest = {}
with open("records.jsonl", encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        latest[record["run_id"]] = record
# 使用 latest.values()，按 run_id 与结构评价表合并。
```

终态为 `succeeded`、`failed` 或 `interrupted`。
`succeeded` 只表示生成和请求的导出/渲染完成，不代表结构有效或物理稳定。
`stage` 标明初始化、生成、导出或渲染阶段，`error` 保存异常类型、消息及堆栈。
单次普通异常不会丢弃该样本或阻止其他组合；初始化失败时记录整批失败，且不反复加载模型。
Ctrl-C 会保存当前实验为 `interrupted` 并停止，未尝试项目仍留在 manifest，汇总中计为 pending。
若进程被强制终止，最后的 `running` 事件用于识别未完成实验；不自动重试或覆盖。
日志每条事件都 flush/fsync。磁盘写入失败无法保证持久化，将停止批次并报错。

`generation_seconds` 包含设置种子与官方生成调用，成功时使用设备同步后计时，
不包含模型初始化、文件导出或渲染。生成异常时它是异常出现前的耗时。
`total_seconds` 包含本次初始化（若首次加载）、生成和导出/渲染，不含日志落盘。
记录中的 `statistics` 来自官方返回的砖块数、拒绝原因及回退次数，不能当作质量评价。

## 温度记录边界

官方每生成一块砖都将温度重置为 `initial_temperature`。
在拒绝采样过程中再次生成已拒绝的候选时，温度可能按
`min(max_temperature, temperature + temperature_increase)` 增加。
因此记录完整的初始温度、升温步长、上限和策略，而不是把整个过程称作固定温度。
默认 logit masking 开启时，官方忽略 `top_k` / `top_p`，记录中的对应 effective 字段为 `null`。
本轮没有插桩逐砖/逐 token 温度轨迹，`actual_temperature_trace` 为 `null`，
`trace_status` 为 `not_instrumented`，不能据此声称已观测最高温度或实际升温次数。
mock 的 effective 配置只模拟策略相关字段；真实模式才从官方 dataclass 展开完整配置。

## 本地测试

```sh
python -S -m unittest discover -s tests -p test_run_experiments.py -v
```

`-S` 排除 site-packages，确保 mock 和调度逻辑不依赖大型库。
只运行这一组测试，避免官方 `tests/test_infer.py` 加载真实模型。
测试覆盖组合展开、唯一 ID、Unicode、重复批次、事件持久化、失败后继续、初始化失败、
部分导出保留、渲染失败、Ctrl-C、温度策略、配置与种子传递及 CLI 返回码。
官方适配器测试仅注入替身，不验证模型推理、GPU、Gurobi 或 Blender 的真实行为。
