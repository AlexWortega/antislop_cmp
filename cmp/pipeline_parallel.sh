#!/bin/sh
# Generation, both judges and Kimi calibration all at once; judges re-pass until generation is done.
cd /mnt/storage/antislop_cmp/code
export EXP=homework TASKS=tasks_homework JUDGE_WORKERS=16
GEN_WORKERS=32 python3 run.py gen-api > log_gen.log 2>&1 &
GEN=$!
JUDGE=moonshotai/kimi-k3 JSUF=_kimi python3 calib.py judge > log_calib_kimi.log 2>&1 &
CAL=$!
judge_loop() {  # judge_loop LOG [env...]: keep judging new outputs while generation runs
  log=$1; shift
  while kill -0 $GEN 2>/dev/null; do env "$@" python3 run.py judge >> "$log" 2>&1; sleep 20; done
  env "$@" python3 run.py judge >> "$log" 2>&1
}
judge_loop log_judge_opus.log X=1 &
J1=$!
judge_loop log_judge_kimi.log JUDGE=moonshotai/kimi-k3 JSUF=_kimi &
J2=$!
wait $GEN $CAL $J1 $J2
python3 run.py report > report_opus.txt 2>&1
JSUF=_kimi python3 run.py report > report_kimi.txt 2>&1
JUDGE=moonshotai/kimi-k3 JSUF=_kimi python3 calib.py report > calib_kimi.txt 2>&1
echo PIPELINE_DONE
