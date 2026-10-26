#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
receiver="$script_dir/../tools/adv_broadcast_realtime.py"

usage() {
    cat <<'EOF'
Usage: demo_usrp_rx.sh [options]

Receive PhantomChannel packets live from BLE advertising channel 39. The
receiver runs until Ctrl-C unless --duration is supplied.

Options:
  --usrp-args ARGS    UHD device arguments
  --sample-rate SPS   Sample rate (default: 4000000)
  --gain DB           RX gain (default: 50)
  --antenna NAME      RX antenna (default: RX2)
  --address ADDR      Only display this advertiser address
  --duration SEC      Stop after SEC; default 0 means continuous
  --once              Exit after the first valid PhantomChannel packet
  --show-duplicates   Print every repeated packet (default suppresses repeats)
  -h, --help          Show this help
EOF
}

usrp_args="type=b200,master_clock_rate=16e6"
sample_rate="4000000"
gain="50"
antenna="RX2"
address=""
duration="0"
extra=()

while (($#)); do
    case "$1" in
        --usrp-args|--sample-rate|--gain|--antenna|--address|--duration)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a value" >&2; exit 2; }
            case "$1" in
                --usrp-args) usrp_args=$2 ;;
                --sample-rate) sample_rate=$2 ;;
                --gain) gain=$2 ;;
                --antenna) antenna=$2 ;;
                --address) address=$2 ;;
                --duration) duration=$2 ;;
            esac
            shift 2
            ;;
        --once|--show-duplicates)
            extra+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command=(
    python3 "$receiver"
    --platform nrf52840
    --observe
    --no-uart
    --source uhd
    --usrp-args "$usrp_args"
    --sample-rate "$sample_rate"
    --gain "$gain"
    --antenna "$antenna"
    --duration-s "$duration"
)
[[ -z "$address" ]] || command+=(--address "$address")
command+=("${extra[@]}")
exec "${command[@]}"
