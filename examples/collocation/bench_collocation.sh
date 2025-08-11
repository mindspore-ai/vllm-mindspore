#!/bin/bash

REQUEST_RATE0=${1:-1.0}      # RPS for server 0
REQUEST_RATE1=${2:-1.0}      # RPS for server 1
TOTAL_REQUEST=${3:-100}

MODEL="/yourpath/Qwen2.5-7B-Instruct"
PROMPT="I love Beijing because"
MAX_TOKENS=256

REQUEST_BODY="{\"model\": \"${MODEL}\", \"prompt\": [\"${PROMPT}\"], \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0, \"top_p\": 1.0, \"top_k\": 5, \"repetition_penalty\": 1.0}"

SERVER_0="http://localhost:24500/v1/completions"
SERVER_1="http://localhost:24501/v1/completions"

INTERVAL_0=$(awk -v r="$REQUEST_RATE0" 'BEGIN{printf "%.3f", 1/r}')
INTERVAL_1=$(awk -v r="$REQUEST_RATE1" 'BEGIN{printf "%.3f", 1/r}')

t_start=$(date +%s.%N)

send_requests () {
    local SERVER="$1"
    local INTERVAL="$2"
    local TOTAL_REQ="$3"
    for ((n=1; n<=TOTAL_REQ; n++)); do
        echo "[$(date '+%H:%M:%S')] send request to -> $SERVER (#$n)"

        curl -s "$SERVER" -H "Content-Type: application/json" -d "$REQUEST_BODY" &

        if (( n < TOTAL_REQ )); then
            sleep "$INTERVAL"
        fi
    done

    wait
}

send_requests "$SERVER_0" "$INTERVAL_0" "$TOTAL_REQUEST" &
send_requests "$SERVER_1" "$INTERVAL_1" "$TOTAL_REQUEST" &

wait

t_end=$(date +%s.%N)
elapsed=$(echo "$t_end - $t_start" | bc -l)
elapsed_format=$(printf "%.2f" "$elapsed")

printf '\n'
printf '%*s\n' 40 '' | tr ' ' '>'
printf 'Collocation: All requests finished.\n -----------Recap-----------\n'
echo "Server 0 Request Rate: $REQUEST_RATE0 RPS"
echo "Server 1 Request Rate: $REQUEST_RATE1 RPS"
echo "Total Requests: $TOTAL_REQUEST"
echo "Total Time Taken: $elapsed_format seconds"
printf '%*s\n' 40 '' | tr ' ' '>'