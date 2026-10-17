| step | rows removed | rows remaining |
|---|---:|---:|
| raw rows | 0 | 3,831,240 |
| missing pickup or dropoff timestamp | 0 | 3,831,240 |
| pickup outside 2026-04 | 12 | 3,831,228 |
| duration < 60s (zero, negative or glitch) | 88,176 | 3,743,052 |
| duration > 3h (multi-hour / multi-day) | 1,249 | 3,741,803 |
| pickup or dropoff zone id not in 1..265 | 0 | 3,741,803 |
