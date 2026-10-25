#!/usr/bin/env bash
# Full AutoDL workflow. All model execution stays in the existing stage scripts.
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

usage() {
  cat <<'HELP'
用法：bash run_all.sh [--from 阶段] [--skip-predict] [--skip-install] [--skip-smoke]

默认依次运行：install → prepare → smoke → train → evaluate → predict
--from 阶段      从 install/prepare/smoke/train/evaluate/predict 开始
--skip-predict   本次完成训练评估后结束，不运行全量测试集
--skip-install   跳过依赖安装（需要已经安装完成）
--skip-smoke     跳过小规模GPU检查（需要此前检查通过）
--help          显示说明，不启动任务

支持原有环境变量：CONFIG、RUN_NAME、PILOT_GROUPS、SAVE_EVERY、RESUME、WARM_START。
日志默认写入 autodl/.cache/run-all/；LOG_DIR 可指定其他日志目录。
HELP
}

START_STAGE=install
SKIP_INSTALL=0
SKIP_SMOKE=0
SKIP_PREDICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
      START_STAGE="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --skip-predict) SKIP_PREDICT=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$START_STAGE" in
  install|prepare|smoke|train|evaluate|predict) ;;
  *) printf '未知起始阶段：%s\n' "$START_STAGE" >&2; exit 2 ;;
esac

export CONFIG="${CONFIG:-configs/autodl.json}"
if [[ "$CONFIG" != /* ]]; then CONFIG="$PROJECT_DIR/$CONFIG"; fi
if [[ ! -f "$CONFIG" ]]; then printf '配置文件不存在：%s\n' "$CONFIG" >&2; exit 2; fi
export RUN_NAME="${RUN_NAME:-pilot${PILOT_GROUPS:-64}}"
export PYTHONUNBUFFERED=1
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/.cache/run-all/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/.cache"

# AutoDL/Linux provides flock. Its descriptor also closes on interruption/exit.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$PROJECT_DIR/.cache/run-all.lock"
  if ! flock -n 9; then
    printf '已有一键任务正在此项目中运行，请查看原任务日志。\n' >&2
    exit 1
  fi
fi

CURRENT_STAGE=启动
on_failure() {
  local code=$?
  trap - ERR
  printf '\n任务在阶段 %s 停止，退出码 %s。日志：%s\n' "$CURRENT_STAGE" "$code" "$LOG_DIR" >&2
  printf '排除问题后可运行：bash run_all.sh --from %s\n' "$CURRENT_STAGE" >&2
  exit "$code"
}
trap on_failure ERR

printf '运行名称：%s\n配置：%s\n日志目录：%s\n' "$RUN_NAME" "$CONFIG" "$LOG_DIR"
stages=(install prepare smoke train evaluate predict)
scripts=(install.sh prepare.sh smoke.sh train_pilot.sh evaluate.sh predict.sh)
started=0
for i in "${!stages[@]}"; do
  stage="${stages[$i]}"
  if [[ "$stage" == "$START_STAGE" ]]; then started=1; fi
  if [[ "$started" == 0 ]]; then continue; fi
  if [[ "$stage" == install && "$SKIP_INSTALL" == 1 ]] ||
     [[ "$stage" == smoke && "$SKIP_SMOKE" == 1 ]] ||
     [[ "$stage" == predict && "$SKIP_PREDICT" == 1 ]]; then
    printf '跳过阶段：%s\n' "$stage"
    continue
  fi
  CURRENT_STAGE="$stage"
  printf '\n[%s/6] 开始 %s\n' "$((i+1))" "$stage"
  bash "$PROJECT_DIR/scripts/${scripts[$i]}" 2>&1 | tee "$LOG_DIR/$((i+1))-$stage.log"
  printf '[%s/6] 完成 %s\n' "$((i+1))" "$stage"
done
trap - ERR
printf '\n本次选定阶段全部完成。日志：%s\n' "$LOG_DIR"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  WORK_DIR="$("$PROJECT_DIR/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' "$CONFIG")"
  if [[ -f "$WORK_DIR/evaluation/$RUN_NAME/comparison.json" ]]; then
    printf '评估结果：%s/evaluation/%s/comparison.json\n' "$WORK_DIR" "$RUN_NAME"
  fi
  if [[ -f "$WORK_DIR/test/$RUN_NAME/rl.csv" ]]; then
    printf '提交目录：%s/test/%s/ （rag.csv / critic.csv / rl.csv）\n' "$WORK_DIR" "$RUN_NAME"
  fi
fi
