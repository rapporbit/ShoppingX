#!/bin/bash
# 第二夜（P0）实验编排。跑在 GPU 机器，nohup 后台跑一夜。
#
# 首夜结论：四组全部超基线，但 hard negative 占比越高主指标越低（v1 16.4% > v2 54.4% ≈ v3 71.2%），
# 与"挖难负例提升效果"的预期相反。对着 refdocs 04-2 复盘，三条嫌疑：
#   1. §5.2 的 hard negative 是**串行课程**（易→难同一模型一路训），我们做成了**平行对照**（各自
#      从 v0 起跑），于是"只上过第一课"的 v1 赢——这不矛盾，是我们把范式做拧了。
#   2. §4.2 的温度是**动态**的：跨语言对 0.02、同语言对 0.05。我们全局钉 0.02，而 ESCI 基本同语言。
#      低温 × 高 hard 占比 = 过度推开近邻，与"complement 压得越狠、recall 也越低"的走势吻合。
#   3. §3.1 高质量正例该重复使用，我们每 query 只取 1 个，扔掉 74% 人工标注 → 2 epoch 就过拟合。
#
# 本轮拆成单变量验证，顺序按「先出最有信息量的结果」排，中途挂掉也不白跑。
set -u
cd ~/globex-train
mkdir -p logs results

export PATH=$HOME/.local/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
# 必须有：评测 base 模型时 transformers 会去校验 huggingface.co，没镜像就是几分钟的静默超时——
# 表现为「脚本在跑但 GPU 0%」，看着像卡死。train_one.sh 里有，这里当初漏了。
export HF_ENDPOINT=https://hf-mirror.com

stamp() { date "+%F %T"; }

xling() {   # xling <模型路径> <结果名>
  echo "[$(stamp)] XLING $2 <- $1"
  uv run python eval_crosslingual.py --model "$1" --pairs xlingual_pairs.jsonl \
    --out results/xling_$2.json > logs/xling_$2.log 2>&1
  echo "[$(stamp)] XLING $2 rc=$? $(grep -o '"pass_rate": [0-9.]*' results/xling_$2.json 2>/dev/null)"
}

evaluate() {   # evaluate <模型路径> <结果名>
  echo "[$(stamp)] EVAL $2 <- $1"
  uv run python eval_on_gpu.py --model "$1" --corpus corpus.jsonl \
    --qrels esci_eval_qrels.jsonl --out results/$2.json > logs/eval_$2.log 2>&1
  echo "[$(stamp)] EVAL $2 rc=$?"
}

ckpt_at() {  # ckpt_at <output_dir> <第几个epoch>：显式挑 checkpoint，不让框架替我们选
  # 必须先锁定**最近一次 run** 的子目录（ms-swift 每跑一次就新建 v0-/v1-/v2- 时间戳目录），
  # 否则重跑时新旧 checkpoint 会混在一起排序，第 N 个取到上一次的产物——静默串台，最难查。
  local run
  run=$(find "$1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V | tail -1)
  [ -z "$run" ] && return
  find "$run" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | sed -n "$2p"
}

run() {   # run <标签> <数据前缀> <epochs> <lr> <temp> [初始模型]
  local TAG=$1 VER=$2 EP=$3 LR=$4 TEMP=$5 INIT=${6:-BAAI/bge-m3}
  echo "[$(stamp)] TRAIN $TAG (ver=$VER ep=$EP lr=$LR temp=$TEMP init=$INIT)"
  VER=$VER TAG=$TAG EPOCHS=$EP LR=$LR TEMP=$TEMP INIT_MODEL=$INIT \
    ./train_one.sh > logs/train_$TAG.log 2>&1
  local C
  C=$(ckpt_at output/$TAG "$EP")
  echo "[$(stamp)] TRAIN $TAG done ckpt=$C"
  [ -n "$C" ] && { evaluate "$C" "$TAG"; xling "$C" "$TAG"; }
}

echo "[$(stamp)] === P0 开始 ==="

# 闸门：先体检中文，秒级。首夜训完的模型中文到底崩没崩，是后面所有排名的前提。
xling BAAI/bge-m3 base
xling "$(ckpt_at output/v1-e2 1)" n1_v1     # 首夜最好的那个 checkpoint

# E7：只改超参（τ 0.02→0.05、lr 1e-5→5e-6），数据仍是首夜的 v1，单变量
run e7_tune v1 2 5e-6 0.05

# E9：严格补首夜缺口——原配置跑满 2 epoch 并保留全部 checkpoint（首夜 save_total_limit=1
#     撞上 best-ckpt 逻辑，v1 的 epoch2 被自动删了，导致对照 epoch 数不齐）
run e9_repro v1 2 1e-5 0.02

# E5：P0 全套装（正例展开 + 新温度 + 新 lr）
run e5_full v1x 2 5e-6 0.05

# E6：只加正例展开，超参保持首夜原样，用来拆出「展开」单独的贡献
run e6_expand v1x 2 1e-5 0.02

# E8（P2 试水）：curriculum——拿 E5 的成品当起点，再喂 hard 占比 71.2% 的 v3 数据训 1 epoch。
#     这才是 refdocs §5.2 说的「先学基础再上 hard」。若它比 E5 更好，就证明首夜的结论只是顺序错了。
CKPT_E5=$(ckpt_at output/e5_full 2)
[ -n "$CKPT_E5" ] && run e8_curric v3 1 5e-6 0.05 "$CKPT_E5"

echo "[$(stamp)] === 全部完成 ==="
for f in results/*.json; do echo "--- $f"; cat "$f"; echo; done
