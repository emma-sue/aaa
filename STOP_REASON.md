# Stop Reason

`NONE` — AIO-3 Stage-A is active on 4 x RTX 4090 in tmux `srsc_pipeline`
with per-GPU batch 30 / global batch 120; the latest
independently verified atomic recovery point is epoch 60 / step 124040 /
batch 0, SHA256
`ab3aa8376e76b31072335269e25424a41b5c6df1e9ee878dc6936cc094179b98`.
The AIO-3 and AIO-5 manifests both have zero missing entries.  The entry is
`scripts/launch_aio3_stage_a_4x4090.sh`, which returns to
`launch_when_data_ready.sh` after Stage-A, retaining Stage-B/Stage-C and AIO-5.
The watchdog and epoch65-240 trend chain are detached in
tmux.  An SSH/network interruption is not a stop condition.
