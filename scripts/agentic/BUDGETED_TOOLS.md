# 预算约束下的专家选择：服务器运行说明

计划与验收项见 `plan.md` 第 14 节。所有命令从 `lime-rec/` 目录运行。
这是独立的缓存回放实验，**真实专家推理和模型训练仅在服务器执行**。

## 1. 协议冻结

- `sasrec`、`itemcf`、`semantic` 每次返回自己的 Top-20 与完整目录归一化后的分数。
  初始状态只有交互历史；调用前没有候选并集、未调用专家排名、分数或 target。
- 预算为 1 或 2 次不同工具调用。成功运行用满预算；非法 JSON/未知或重复工具均计失败，
  输出空排名，不自动回退。网络/模型服务错误中断运行，不能将未完成目录作为正式结果。
- 所有策略的最终排名规则相同：被调用工具的固定权重 0.60/0.15/0.25 重归一化，
  未返回候选的该路分数记零，历史项减 0.10。没有调用的专家不贡献分数。
- 简单 router：validation 拟合线性 ridge，预测下一步调用后的 NDCG@10；历史长度、
  重复比例、剩余预算，以及**已观察**分数的统计量作为输入。固定 ridge=1；
  不根据 test 挑参数。它是一步贪心策略，不是最优多步规划。
- SFT：同一 validation 状态下，用真实 validation target 计算各动作的一步效用，
  唯一最优工具作标签；并列状态跳过。prompt 与正式调用字节一致。test 不参与拟合。
- 三工具全调用是预算 3 的参考；它不是 1/2 调用预算下的公平成本对手，也不是原文 fitted gate。
- 不同策略揭示的候选集可以不同；用户/目标相同。目标不在任何返回列表的用户仍计入分母。

## 2. 新建缓存并核验（各域、各 seed）

以下模板默认 seed0、1000 名测试用户。正式样本数与模型选择需事先确定。
先用单独目录、少量用户验证运行流程，再建立正式目录；已有文件/目录不会被覆盖。
模型路径沿用现有 workflow；运行前确认对应文件存在。服务器历史路径见 `plan.md`，
这里不假定当前机器已具备相同部署。

```bash
# 已激活项目 Python 环境，并位于 lime-rec/
python -m pytest -q tests/test_budgeted_tools.py

RUN_ROOT=outputs/budgeted_tools_v1
TRAIN_SEED=0
TEST_USERS=1000
for DOMAIN in beauty toys sports; do
  EXP="$RUN_ROOT/amazon_${DOMAIN}/seed${TRAIN_SEED}"
  test ! -e "$EXP" || { echo "Output exists: $EXP"; exit 1; }
  mkdir -p "$EXP/cache"
  SAS="outputs/models/controlled_gram/amazon_${DOMAIN}_sasrec_max20_seed${TRAIN_SEED}.pt"
  CF="output_final/models/amazon_${DOMAIN}_itemcf.json"
  EMB="output_final/models/amazon_${DOMAIN}_bge_base.npz"
  test -f "$SAS" && test -f "$CF" && test -f "$EMB" || exit 1
  for SPLIT in validation test; do
    LIMIT=()
    if [ "$SPLIT" = test ]; then LIMIT=(--max-users "$TEST_USERS"); fi
    python -m scripts.agentic.build_candidate_cache \
      --config "configs/amazon_${DOMAIN}.json" \
      --sasrec-model "$SAS" --itemcf-model "$CF" --semantic-emb "$EMB" \
      --split "$SPLIT" --candidate-per-expert 20 --sample-seed 2027 \
      --device cuda "${LIMIT[@]}" --out "$EXP/cache/$SPLIT.jsonl" || exit 1
    python -m scripts.agentic.run_budgeted_tools prepare \
      --config "configs/amazon_${DOMAIN}.json" \
      --candidate-cache "$EXP/cache/$SPLIT.jsonl" --split "$SPLIT" \
      --device cuda --out "$EXP/$SPLIT.json" || exit 1
  done
  python -m scripts.agentic.run_budgeted_tools fit \
    --data "$EXP/validation.json" --out-dir "$EXP/fitted" || exit 1
  python -m scripts.agentic.run_budgeted_tools evaluate \
    --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
    --out-dir "$EXP/baselines" || exit 1
 done
```

上面的 shell 使用 Bash 数组，保存为脚本后用 `bash` 执行。
后续用 `TRAIN_SEED=1`、`2` 的独立目录重复；每次从 `RUN_ROOT` 起执行完整代码段并修改 seed。
也可以直接提供已有 candidate cache：`prepare` 会实际重算专家分数、核验候选池，
并记录模型字节哈希、数据哈希和原始缓存哈希；数值差异超出 `rtol=1e-5, atol=1e-6` 会报错。
不要用不同专家的 validation/test 缓存拼成一次实验。

## 3. 训练选择器 LoRA

新的任务是选择工具；原来的排名 SFT checkpoint 不能直接当作已经适配的选择器。
对每个域/seed 使用其 `selector_sft.jsonl`，沿用现有训练与 merge 脚本：

```bash
EXP=outputs/budgeted_tools_v1/amazon_beauty/seed0
BASE_MODEL=/absolute/path/to/Qwen3-4B
# 这两个输出必须尚不存在；具体 GPU 按服务器分配设置。
test ! -e "$EXP/selector_adapter" && test ! -e "$EXP/selector_merged" || exit 1
# 确认 fit_manifest.json 的 sft_examples > 0，max-examples=0 使用全部导出样本。
CUDA_VISIBLE_DEVICES=0 python -m scripts.agentic.train_adapted_controller \
  --model "$BASE_MODEL" --data "$EXP/fitted/selector_sft.jsonl" \
  --out "$EXP/selector_adapter" --max-examples 0 --epochs 1 --seed 2027
python -m scripts.agentic.merge_adapted_controller \
  --base "$BASE_MODEL" --adapter "$EXP/selector_adapter" --out "$EXP/selector_merged"
```

SFT 依赖与既有 trainer 相同：PyTorch、transformers、peft；需要已有服务器训练环境。
运行前检查 `fit_manifest.json`。不要根据正式 test 比较结果选择 epoch、模型或 prompt。
训练及部署使用 `enable_thinking=False`，与现有 Qwen3 SFT 的 chat template 一致。

## 4. 测试实际模型服务

在服务器按既有 vLLM/OpenAI-compatible 部署流程分别服务原始模型和新选择器模型。
Qwen3 服务需配置 chat-template `enable_thinking=false`，否则可能把预算耗在未训练的
思考输出上。`LLM_MODEL` 必须与服务端模型名一致。每个模型使用独立结果目录。
以下命令会真实调用端点，不使用响应缓存。

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=local
export LLM_MODEL=your-served-selector-model-name
EXP=outputs/budgeted_tools_v1/amazon_beauty/seed0
python -m scripts.agentic.run_budgeted_tools evaluate \
  --data "$EXP/test.json" --router "$EXP/fitted/router.json" \
  --llm --max-tokens 128 --bootstrap-resamples 10000 --seed 2027 \
  --llm-artifact "$EXP/selector_merged/config.json" \
  --out-dir "$EXP/adapted_llm"
```

Prompt-only 模型也使用 `--llm`，指向原始模型端点，指定原始模型配置/训练清单作为
`--llm-artifact`，输出到 `prompt_only_llm`。该参数只记录所提供文件的哈希，
不是对远端权重身份的自动证明；部署侧应另存基座版本、adapter/checkpoint 哈希及启动命令。
不同域适配模型必须与该域的 validation SFT 数据对应。两种模型必须使用相同 token 上限。
`evaluate` 内部同时运行固定策略和 router，所有逐用户结果都来自同一份 test artifact。

## 5. 结果与解释

- `manifest.json`：输入/代码/协议/模型标识、随机种子、计时范围及完成标记名称。
- `COMPLETED.json`：完整成功标记；缺失则该运行不完整，不能据部分输出汇报结果。
- 每个条件的 JSONL：目标仅在策略执行后附加；保存所有用户、可见状态、动作、
  最终排名、失败原因、token、请求耗时。日志含用户历史，请按实验数据权限保管。
- `summary.json`：R/N@5/10、已揭示候选的 target coverage、平均实际调用数、失败率、
  tokens、`mean_cache_replay_wall_seconds`、`mean_llm_request_seconds`。
- `paired_comparisons.json`：各固定序列/router 减去相同预算 LLM 的 R@10/N@10 差值、
  paired-user 双侧 95% CI 和单侧 95% LCB。10,000 次重采样；逐项探索性区间，不做多重校正。

同一预算下固定序列的不同排列也完整报告；固定融合的最终结果对排列不敏感，
保留它们用于核查顺序是否被实现意外引入。不能挑 test 上最好的固定序列当作验证集选择。
LLM 的两次调用可根据第一次结果改变选择，路由器也读取同样可见状态。
当前没有主动提前停止策略：合法轨迹使用完整预算。

缓存回放总耗时包含 Python 调度、读取、融合及 LLM 请求；专家原始推理在准备阶段发生。
**这些时间不能构成等硬件线上加速比。** 多域/多 seed 结果应逐格报告；paired-user CI
条件于当前训练模型，不代表训练 seed 分布的不确定性。跨 seed 汇总应保留逐 seed 结果。
