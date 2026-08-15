#!/usr/bin/env bash
# AetherForge Desktop Console — full build launcher
# Connect Vast.ai / RunPod / SSH → sync → start training → dashboard
set -euo pipefail

ROOT="${AETHERFORGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT/.venv/bin:$HOME/.local/bin:$PATH"
export TERM="${TERM:-xterm-256color}"

DASH_HOST="${AETHERFORGE_DASH_HOST:-127.0.0.1}"
DASH_PORT="${AETHERFORGE_DASH_PORT:-8765}"
DASH_URL="http://${DASH_HOST}:${DASH_PORT}/"
PID_FILE="${XDG_RUNTIME_DIR:-/tmp}/aetherforge-dashboard.pid"
LOG_FILE="${XDG_CACHE_HOME:-$HOME/.cache}/aetherforge-dashboard.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

# ── recipes ──────────────────────────────────────────────────────────
# Flash-0731 specialist (ESFT-LoRA on Vast multi-GPU)
CFG_FLASH=( -c configs/base.yaml -c configs/deepseek_v4_flash.yaml -c recipes/flagship_flash_domain.yaml )
# Public Aether product corpus on Flash
CFG_FLASH_AETHER=( -c configs/base.yaml -c configs/deepseek_v4_flash.yaml -c recipes/flagship_flash_domain.yaml
  -o "data.curated_path=$HOME/aether_data/training/public-aether-dataset.chat.jsonl"
  -o "data.eval_path=$HOME/aether_data/training/public-aether-eval.jsonl"
  -o "data.domain=aether_public"
  -o "data.synthetic.enabled=false"
  -o "run.name=flagship-flash-0731-aether-public" )
# Broad multi-sector / multi-domain (recommended for general capability on 2×96GB)
CFG_FLASH_BROAD=( -c configs/base.yaml -c configs/deepseek_v4_flash.yaml -c recipes/broad_flash_192gb.yaml )
# Wide lattice LoRA (still PEFT — not full-weight bf16)
CFG_FLASH_WIDE=( -c configs/base.yaml -c configs/deepseek_v4_flash.yaml -c recipes/wide_flash_192gb.yaml )
# A3B logistics flagship
CFG_A3B=( -c configs/base.yaml -c configs/qwen_a3b.yaml -c recipes/flagship_logistics_a3b.yaml )
# Qwen3.8-27B dense PEFT (Vast-only live; local = dry-run)
CFG_QWEN38=( -c configs/base.yaml -c configs/qwen38_27b.yaml -c recipes/qwen38_27b.yaml )

C_CYAN=$'\033[36m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_RED=$'\033[31m'
C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'

banner() {
  clear 2>/dev/null || true
  cat <<EOF
${C_CYAN}${C_BOLD}
  ╔══════════════════════════════════════════════════════════════╗
  ║              A E T H E R F O R G E  ·  D E S K T O P             ║
  ║     MoE post-training · Expert Studio · Vast.ai train          ║
  ╚════════════════════════════════════════════════════════════════╝
${C_RST}${C_DIM}  root: $ROOT
  dash: $DASH_URL
  flash weights: $HOME/Downloads/LLM's/DeepSeek-V4-Flash-0731
${C_RST}
EOF
}

need_py() {
  if [[ ! -x "$PY" ]]; then
    echo "${C_YEL}Creating AetherForge venv…${C_RST}"
    cd "$ROOT"
    python3 -m venv .venv
    .venv/bin/pip install -q -U pip wheel
    .venv/bin/pip install -q -e ".[dev]" || .venv/bin/pip install -q -e .
  fi
  cd "$ROOT"
}

hf() {
  "$PY" -m aetherforge.cli "$@"
}

dash_running() {
  curl -sf -m 1 "http://${DASH_HOST}:${DASH_PORT}/api/health" >/dev/null 2>&1
}

start_dashboard() {
  if dash_running; then
    echo "${C_GRN}✓ Dashboard already up${C_RST}  $DASH_URL"
    return 0
  fi
  echo "${C_CYAN}→ Starting Training Console on :${DASH_PORT}…${C_RST}"
  cd "$ROOT"
  nohup "$PY" -m aetherforge.cli dashboard --host "$DASH_HOST" --port "$DASH_PORT" \
    >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  for i in $(seq 1 30); do
    dash_running && break
    sleep 0.25
  done
  if dash_running; then
    echo "${C_GRN}✓ Dashboard live${C_RST}  $DASH_URL  (pid $(cat "$PID_FILE" 2>/dev/null || echo '?'))"
  else
    echo "${C_RED}✗ Dashboard failed — see $LOG_FILE${C_RST}"
    return 1
  fi
}

open_browser() {
  start_dashboard || true
  if command -v google-chrome >/dev/null 2>&1; then
    google-chrome --app="$DASH_URL" \
      --window-size=1600,1000 \
      --user-data-dir="${XDG_CACHE_HOME:-$HOME/.cache}/aetherforge-chrome" \
      --disable-features=TranslateUI \
      >/dev/null 2>&1 &
  else
    xdg-open "$DASH_URL" >/dev/null 2>&1 &
  fi
}

pause() {
  echo
  read -r -p "Press Enter to continue… " _
}

show_status() {
  echo "${C_BOLD}── Connection status ──${C_RST}"
  hf connect status 2>/dev/null || true
  echo
  if dash_running; then
    echo "${C_GRN}Dashboard:${C_RST} up  $DASH_URL"
  else
    echo "${C_YEL}Dashboard:${C_RST} down"
  fi
  if [[ -d "$HOME/Downloads/LLM's/DeepSeek-V4-Flash-0731" ]]; then
    local n
    n=$(ls "$HOME/Downloads/LLM's/DeepSeek-V4-Flash-0731"/model-*-of-00048.safetensors 2>/dev/null | wc -l)
    echo "${C_GRN}Flash-0731 local:${C_RST} $n/48 shards under Downloads/LLM's/DeepSeek-V4-Flash-0731"
  else
    echo "${C_YEL}Flash-0731 local:${C_RST} not found (remote train will pull from Hugging Face)"
  fi
}

save_vast_key() {
  echo "${C_BOLD}Vast.ai API key${C_RST}"
  echo "  Create at: https://cloud.vast.ai/account/ → API keys"
  echo "  Or export VAST_API_KEY=… beforehand and choose 'from env'."
  echo
  echo "  1) Paste key now"
  echo "  2) Load from env (VAST_API_KEY / VASTAI_API_KEY)"
  echo "  0) Cancel"
  read -r -p "Choice: " c
  case "$c" in
    1)
      read -r -s -p "API key: " key
      echo
      [[ -z "$key" ]] && { echo "Empty key."; return 1; }
      hf connect key vast --value "$key"
      ;;
    2)
      hf connect key vast --from-env || {
        echo "${C_RED}No VAST_API_KEY in environment.${C_RST}"
        return 1
      }
      ;;
    *) return 0 ;;
  esac
  echo "${C_GRN}Key saved to ~/.aetherforge/credentials.yaml${C_RST}"
}

list_vast_instances() {
  echo "${C_BOLD}── Vast instances (API) ──${C_RST}"
  "$PY" - <<'PY'
from aetherforge.providers.compute.vast import VastComputeProvider
import json
v = VastComputeProvider()
if not v.api_key():
    print("No API key. Use menu → Save Vast API key first.")
    raise SystemExit(1)
rows = v.list_instances()
if not rows:
    print("No instances visible (rent one on cloud.vast.ai, then refresh).")
else:
    for i, r in enumerate(rows, 1):
        print(f"  [{i}] id={r.get('id')}  status={r.get('status')}  gpu={r.get('gpu')}")
        print(f"      ssh={r.get('ssh_host')}:{r.get('ssh_port')}  label={r.get('label')}")
    print(json.dumps({"count": len(rows)}, indent=2))
PY
}

connect_vast_manual() {
  echo "${C_BOLD}Connect Vast.ai instance (SSH from Vast dashboard)${C_RST}"
  echo "  On cloud.vast.ai → your instance → copy SSH host + port"
  echo "  Example: ssh -p 12345 root@123.45.67.89"
  echo
  read -r -p "SSH host (IP or hostname): " host
  read -r -p "SSH port [22]: " port
  port="${port:-22}"
  read -r -p "User [root]: " user
  user="${user:-root}"
  read -r -p "Identity file path (blank = default ~/.ssh): " ident
  read -r -p "Remote dir [/workspace/aetherforge]: " rdir
  rdir="${rdir:-/workspace/aetherforge}"
  [[ -z "$host" ]] && { echo "Host required."; return 1; }

  args=( connect vast --host "$host" --port "$port" --user "$user" --remote-dir "$rdir" )
  [[ -n "${ident:-}" ]] && args+=( --identity "$ident" )
  echo
  echo "${C_CYAN}→ Testing SSH + registering connection…${C_RST}"
  if hf "${args[@]}"; then
    echo "${C_GRN}✓ Connected${C_RST}"
  else
    echo "${C_RED}✗ Connect failed — check host/port/key and instance is running${C_RST}"
    return 1
  fi
}

connect_vast_from_list() {
  list_vast_instances || return 1
  echo
  read -r -p "Pick instance number (or 0 cancel): " n
  [[ "$n" == "0" || -z "$n" ]] && return 0
  "$PY" - <<PY
from aetherforge.providers.compute.vast import VastComputeProvider
from aetherforge.providers import connect as conn
import sys
v = VastComputeProvider()
rows = v.list_instances()
idx = int("$n") - 1
if idx < 0 or idx >= len(rows):
    print("Invalid index"); sys.exit(1)
r = rows[idx]
host = r.get("ssh_host") or ""
port = r.get("ssh_port") or 22
# ports sometimes nested
if isinstance(port, dict):
    port = port.get("HostPort") or port.get("host_port") or 22
try:
    port = int(str(port).split("/")[0] or 22)
except Exception:
    port = 22
if not host:
    print("Instance has no ssh_host yet — wait until it is running, or paste host manually.")
    sys.exit(2)
print(f"Connecting {host}:{port} …")
res = conn.connect_compute("vast", host=host, port=port, user="root",
                           instance_id=str(r.get("id") or ""), test=True)
import json
print(json.dumps(res, indent=2, default=str))
sys.exit(0 if res.get("ok") else 2)
PY
}

pick_recipe() {
  echo "${C_BOLD}Training recipe${C_RST}"
  echo "  1) Flash-0731 domain specialist (configs/deepseek_v4_flash + flagship_flash_domain)"
  echo "  2) Flash-0731 + public Aether dataset (product corpus)"
  echo "  3) Qwen A3B logistics flagship"
  echo "  4) Qwen3.8-27B dense PEFT (Vast live / local dry-run)"
  echo "  5) Custom (enter -c paths yourself next)"
  read -r -p "Choice [1]: " r
  r="${r:-1}"
  case "$r" in
    1) RECIPE_NAME="flash-domain"; RECIPE_ARGS=( "${CFG_FLASH[@]}" ) ;;
    2) RECIPE_NAME="flash-aether"; RECIPE_ARGS=( "${CFG_FLASH_AETHER[@]}" ) ;;
    3) RECIPE_NAME="a3b-logistics"; RECIPE_ARGS=( "${CFG_A3B[@]}" ) ;;
    4) RECIPE_NAME="qwen38-27b"; RECIPE_ARGS=( "${CFG_QWEN38[@]}" ) ;;
    5)
      RECIPE_NAME="custom"
      read -r -p "Extra aetherforge train args (e.g. -c configs/base.yaml -c …): " custom
      # shellcheck disable=SC2206
      RECIPE_ARGS=( $custom )
      ;;
    *) RECIPE_NAME="flash-domain"; RECIPE_ARGS=( "${CFG_FLASH[@]}" ) ;;
  esac
  echo "Selected: ${C_CYAN}$RECIPE_NAME${C_RST}"
}

ensure_recipe() {
  if [[ -z "${RECIPE_NAME:-}" ]]; then
    pick_recipe
  fi
}

do_remote_plan() {
  ensure_recipe
  echo "${C_CYAN}→ Remote plan (no spend)…${C_RST}"
  hf remote plan "${RECIPE_ARGS[@]}"
}

do_remote_sync() {
  echo "${C_CYAN}→ Rsync project to GPU box…${C_RST}"
  hf remote sync
}

do_remote_launch() {
  ensure_recipe
  echo
  echo "${C_YEL}${C_BOLD}⚠  This will rsync AetherForge to your Vast box and start training.${C_RST}"
  echo "    Background mode (default): nohup on the instance — costs GPU \$ until done."
  echo "    Recipe: $RECIPE_NAME"
  echo "    Args:   ${RECIPE_ARGS[*]}"
  echo
  read -r -p "Type YES to launch: " conf
  [[ "$conf" == "YES" ]] || { echo "Aborted."; return 0; }
  echo "${C_CYAN}→ Sync + remote launch (background)…${C_RST}"
  hf remote launch --exec "${RECIPE_ARGS[@]}"
  echo
  echo "${C_GRN}Launch request finished.${C_RST}"
  echo "  • Logs on box:  artifacts/remote_train.nohup.log"
  echo "  • Pull later:   menu → Pull artifacts"
  echo "  • Tail logs:    menu → Remote logs"
  echo "  • Dashboard:    $DASH_URL"
}

do_pull() {
  echo "${C_CYAN}→ Pulling remote artifacts…${C_RST}"
  hf remote pull
  hf remote logs --tail 60 || true
}

do_remote_logs() {
  hf remote logs --tail 100
}

do_local_dry() {
  ensure_recipe
  echo "${C_CYAN}→ Local dry-run (no GPU, no download)…${C_RST}"
  hf train "${RECIPE_ARGS[@]}" --dry-run
}

do_forensics() {
  echo "${C_BOLD}Sector forensics${C_RST}"
  echo "  1) Flash-0731 12 sectors"
  echo "  2) A3B 8 sectors"
  echo "  3) Qwen3.8-27B dense (no expert sectors — capacity card only)"
  read -r -p "Choice [1]: " c
  c="${c:-1}"
  if [[ "$c" == "2" ]]; then
    hf forensics --family qwen_a3b --num-groups 8 --markdown | less -R
  elif [[ "$c" == "3" ]]; then
    hf forensics --family qwen38_dense --model Qwen/Qwen3.8-27B --num-groups 1 --markdown | less -R
  else
    hf forensics --family deepseek_v4_flash --model deepseek-ai/DeepSeek-V4-Flash-0731 \
      --num-groups 12 --markdown | less -R
  fi
}

do_validate_flash() {
  echo "${C_CYAN}→ validate-flash…${C_RST}"
  hf validate-flash
}

do_doctor() {
  hf doctor
}

main_menu() {
  while true; do
    banner
    show_status
    echo
    echo "${C_BOLD}── Menu ──${C_RST}"
    echo "  ${C_CYAN}1${C_RST})  Open Training Console (browser)"
    echo "  ${C_CYAN}2${C_RST})  Save Vast.ai API key"
    echo "  ${C_CYAN}3${C_RST})  List Vast instances (API)"
    echo "  ${C_CYAN}4${C_RST})  Connect Vast — paste SSH host:port"
    echo "  ${C_CYAN}5${C_RST})  Connect Vast — pick from API list"
    echo "  ${C_CYAN}6${C_RST})  Choose training recipe"
    echo "  ${C_CYAN}7${C_RST})  Remote plan (preview sync + train cmd)"
    echo "  ${C_CYAN}8${C_RST})  Sync code to GPU box"
    echo "  ${C_GRN}${C_BOLD}9${C_RST})  ${C_GRN}${C_BOLD}START TRAINING on Vast${C_RST}  (sync + nohup launch)"
    echo "  ${C_CYAN}10${C_RST}) Pull artifacts + logs from Vast"
    echo "  ${C_CYAN}11${C_RST}) Tail remote logs"
    echo "  ${C_CYAN}12${C_RST}) Local dry-run pipeline"
    echo "  ${C_CYAN}13${C_RST}) Sector forensics"
    echo "  ${C_CYAN}14${C_RST}) validate-flash (Flash-0731 stack)"
    echo "  ${C_CYAN}15${C_RST}) Doctor (deps)"
    echo "  ${C_CYAN}0${C_RST})  Quit"
    echo
    read -r -p "Select: " choice
    case "$choice" in
      1) open_browser; pause ;;
      2) save_vast_key; pause ;;
      3) list_vast_instances; pause ;;
      4) connect_vast_manual; pause ;;
      5) connect_vast_from_list; pause ;;
      6) pick_recipe; pause ;;
      7) do_remote_plan; pause ;;
      8) do_remote_sync; pause ;;
      9) do_remote_launch; pause ;;
      10) do_pull; pause ;;
      11) do_remote_logs; pause ;;
      12) do_local_dry; pause ;;
      13) do_forensics ;;
      14) do_validate_flash; pause ;;
      15) do_doctor; pause ;;
      0|q|Q) echo "Bye."; exit 0 ;;
      *) echo "Unknown"; sleep 1 ;;
    esac
  done
}

# ── entry ────────────────────────────────────────────────────────────
need_py
# Always bring dashboard up when launched from desktop
start_dashboard || true
# Optional: open browser immediately unless --no-browser
if [[ "${1:-}" != "--no-browser" && "${1:-}" != "--menu-only" ]]; then
  open_browser || true
fi
# Auto-start menu; allow: aetherforge-desktop --train for jump
if [[ "${1:-}" == "--train" ]]; then
  pick_recipe
  do_remote_launch
  pause
fi
main_menu
