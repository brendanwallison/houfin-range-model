# TACC allocation-time audit, November 3

Scope: November 3, 01:11:11-06:03:59 CST, from the launch of the persistent local worker through expiration of the existing A100-small allocation. This is not an audit of every hour of the allocation or a full GPU-utilization history.

The independent cell-year abundance-correction and private-field decoupling experiments were not implemented or queued. The overnight worker processed the older validated experiment backlog. An operational queue and active workers did not establish readiness of those two scientific experiments. The later explanation about treating infrastructure as completion did not account for the false queue/readiness status communicated to the user.

## What the records establish

| Category | Measured amount | Interpretation |
|---|---:|---|
| Calculated allocation cost in this window | 9.7008 SUs | Running the older queue, including the overhead below; not spending on the two missing experiments |
| Paid GPU-capacity overhead identifiable from phase/worker records | approximately .7248 SUs | CPU-only post-fit diagnostics while GPUs remained allocated, plus unused development GPU lanes after their tasks ended |
| Unsaved interval before the final allocation timeout | 9m15s, approximately .2313 SUs | Last checkpoint retained step 200; the log reached step 281. This interval will need repeating on resume, apart from any reusable compilation effects |
| Paid empty-queue waiting caused by the missing experiments | No such interval identified | The small worker continued taking older work; this is not a claim of perfect GPU utilization |
| Requested development time released early | 1h12m23s | Unallocated after early completion, hence not charged |

The identifiable GPU-capacity overhead consists of 1,477.56 seconds of single-A100 allocation time on CPU-only post-fit checks, plus 209.67 GPU-seconds of CPU-only checks and 183.17 GPU-seconds outside task execution across the three-GPU development allocation. These CPU diagnostics were useful work, but did not require reserving those GPUs. Marker timestamps have approximately one-second resolution. This is a phase-based estimate, not a continuous utilization measurement; other GPU idle periods within fitting/diagnostics are not quantified.

The approximately .7248 SU overhead and .2313 SU unsaved interval are disjoint, totaling approximately .9560 SUs of identified capacity overhead/interrupted progress. This is not a definitive total of all wasted computation, nor evidence that the remaining older-experiment work had no scientific value.

## Slurm evidence

- A100-small job 3544518: started November 1 at 06:03:32; TIMEOUT November 3 at 06:03:59. Worker logs show consecutive older tasks throughout the audited window. The final task started at 05:31:10 and was still performing MAP optimization at timeout.
- Three-GPU development job 3548096: started November 3 at 01:16:23; COMPLETED at 02:04:00, after 2,857 seconds. All three GPU lanes ran separate older registered departure fits. Their task durations were approximately 2,739, 2,794 and 2,855 seconds.
- Regular jobs 3547292 and 3548331 were pending with zero allocated nodes and zero elapsed allocation time in the captured Slurm record.
- After 06:03:59, no TACC GPU allocation was active in the captured 06:27:37 snapshot. That gap was unallocated, not billed idle time. The local 3090 continued working.

The resource watcher's condition had also allowed pending regular requests to block another development request after 02:04. This delayed requesting replacement capacity; it did not hold an idle node. It was corrected in the preceding progress turn. New development request 3548968 and small request 3548970 were pending at the captured snapshot. No new allocation is requested by this audit.

## Calculation and limits

The audited portion of the small allocation costs (17,568 seconds / 3,600) * 1.5 = 7.32 SUs. The development allocation costs (2,857 / 3,600) * 3 = 2.380833 SUs. Sum: 9.700833 SUs. These are calculated from Slurm times and [TACC's published charging rules](https://docs.tacc.utexas.edu/hpc/6lonestar/running/), not an itemized accounting invoice. Pending requests and released time do not incur node-hour charges. The minimum-charge rule does not affect these jobs because both exceeded fifteen minutes.

Machine-readable result: [ALLOCATION_TIME_AUDIT_20261103.json](ALLOCATION_TIME_AUDIT_20261103.json). Supporting phase/checkpoint snapshots are preserved in `.research_runtime/ALLOCATION_PHASE_AUDIT_20261103.json` and `ALLOCATION_CHECKPOINT_AUDIT_20261103.json`; calculation code is `.research_runtime/calculate_allocation_audit.py`.

The preceding final calculation attempt was interrupted because automatic approval review could not complete after a usage-limit error. The command did not execute then. After the user requested resume, the same calculation was approved and completed; no approval check was bypassed.
