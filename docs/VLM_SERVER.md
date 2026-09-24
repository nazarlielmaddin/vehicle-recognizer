# Qwen2.5-VL server deploy (Ubuntu server, limitsiz CPU/RAM)

## Variant A — vLLM (tövsiyə, GPU varsa saniyələr, CPU-da dəqiqələr)
```bash
pip install vllm qwen-vl-utils
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8001 \
  --max-model-len 4096 --gpu-memory-utilization 0.9
# CPU-only server: vllm serve ... --device cpu (yavaş, amma işləyir)
```
Model seçimi (RAM, fp16): 3B ~8GB / 7B ~16GB / 32B ~70GB. int4 (AWQ) ilə
təxminən yarıya enir. Dövlət radarı üçün 7B balans, maksimum dəqiqlik üçün 32B.

## Variant B — transformers direct (bu repodakı `local` rejim)
`vlm_server.mode: local`, model ID-i konfiqdə dəyiş. 7B üçün ~16GB RAM.

## Aktivləşdirmə
`configs/default.yaml`:
```yaml
vlm_server:
  enabled: true
  mode: remote
  endpoint: "http://<server-ip>:8001/v1"
```
Serveri yenidən başlat. UI-də `🧠 Qwen2.5-VL:` sətri görünəcək.
QEYD: Qwen License (Apache deyil) — production-dan əvvəl hüquqi rəy şərtdir.
