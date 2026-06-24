#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
PID_FILE="logs/refgrader.pid"

usage() {
    cat <<EOF
Usage: $0 <action> [options]

Actions:
  run       使用 nohup 启动实验 (默认)
  stop      发送 SIGTERM 优雅停止实验
  status    检查实验是否在运行
  tail      实时查看最新日志
  restart   先 stop 再 run

Options (for run):
  所有 run 之后的参数会原样传递给 main_pipeline.py

Examples:
  $0 run --mode FULL --questions Q5 Q6 Q7
  $0 run --mode VARIANCE_OPT --questions Q1 Q2 --sample-size 5
  $0 run --mode FULL --questions Q4 --img-limit 10 --force-rerun
  $0 stop
  $0 status
  $0 tail
EOF
}

get_log_file() {
    if [ -f "$PID_FILE" ]; then
        sed -n '1p' "$PID_FILE"
    else
        ls -t "$LOG_DIR"/experiment_*.log 2>/dev/null | head -1
    fi
}

case "${1:-run}" in
    run)
        shift || true
        RUN_ID="$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$LOG_DIR"
        LOG_FILE="$LOG_DIR/experiment_${RUN_ID}.log"
        export REFGRADER_ARTIFACT_RUN_ID="$RUN_ID"

        echo "[$RUN_ID] 正在启动 RefGrader 实验..."
        echo "  日志文件: $LOG_FILE"

        nohup bash -c '
            child_pid=""
            terminate_child() {
                if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
                    kill -TERM "$child_pid" 2>/dev/null || true
                    wait "$child_pid" 2>/dev/null || true
                fi
                exit 143
            }
            trap terminate_child TERM INT

            python3 main_pipeline.py \
                --run-id "$1" \
                --log-dir "$2" \
                "${@:3}" &
            child_pid=$!
            wait "$child_pid"
            status=$?

            if [ "$status" -eq 0 ] && [ -n "${REFGRADER_POST_SUCCESS_CMD:-}" ]; then
                echo ""
                echo "Main pipeline completed successfully. Running post-success command:"
                echo "  $REFGRADER_POST_SUCCESS_CMD"
                bash -c "$REFGRADER_POST_SUCCESS_CMD" &
                child_pid=$!
                wait "$child_pid"
                status=$?
                child_pid=""
            fi
            trap - TERM INT
            exit "$status"
        ' _ "$RUN_ID" "$LOG_DIR" "$@" \
            >> "$LOG_FILE" 2>&1 &

        PID=$!
        echo "$LOG_FILE" > "$PID_FILE"
        echo "$PID" >> "$PID_FILE"

        echo "  进程 PID: $PID"
        echo ""
        echo "  管理命令:"
        echo "    $0 stop      # 优雅停止"
        echo "    $0 status    # 查看状态"
        echo "    $0 tail      # 实时日志"
        echo "    python monitor.py --watch   # 进度监控"
        ;;

    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "未找到 PID 文件，实验可能未在运行。"
            exit 1
        fi
        PID=$(sed -n '2p' "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "正在发送 SIGTERM 到进程 $PID..."
            kill -TERM "$PID"
            echo "等待优雅关闭 (最多 60 秒)..."
            for i in $(seq 1 60); do
                if ! kill -0 "$PID" 2>/dev/null; then
                    echo "进程已退出。"
                    rm -f "$PID_FILE"
                    exit 0
                fi
                sleep 1
            done
            echo "进程 60 秒内未退出，发送 SIGKILL..."
            kill -KILL "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
        else
            echo "进程 $PID 不在运行，清理 PID 文件。"
            rm -f "$PID_FILE"
        fi
        ;;

    status)
        if [ ! -f "$PID_FILE" ]; then
            echo "实验未在运行 (无 PID 文件)。"
            exit 0
        fi
        PID=$(sed -n '2p' "$PID_FILE")
        LOG=$(sed -n '1p' "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "实验运行中: PID=$PID"
            echo "  日志文件: $LOG"
            echo "  进度文件: results_rrd_vlm/progress.json"
            echo ""
            echo "  查看进度: python monitor.py --watch"
            echo "  查看日志: $0 tail"
        else
            echo "过期 PID 文件 (进程 $PID 不在运行)。"
            rm -f "$PID_FILE"
        fi
        ;;

    tail)
        LOG_FILE="$(get_log_file)"
        if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
            echo "正在追踪日志: $LOG_FILE"
            echo "---"
            tail -f "$LOG_FILE"
        else
            echo "未找到日志文件。"
            exit 1
        fi
        ;;

    restart)
        "$0" stop
        sleep 2
        shift || true
        "$0" run "$@"
        ;;

    *)
        usage
        exit 1
        ;;
esac
