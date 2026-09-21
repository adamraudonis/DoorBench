# Local recorded-contact Jev experiment

The frozen study at `out/local-jev/contact-adjustment-study-001` asks Jev
`jev-1.13.0` about current finger load, individual thumb/finger opposition,
continued qualified grip over the next 100 ms, and small angle candidates.
Requests contain actual contact histories and unstepped geometric probes.
Expected labels and future outcomes remain evaluator-only. These are
privileged simulator measurements, not camera or tactile-only perception.

The first ten-call batch returned eight HTTP 529 errors and two timeouts.
The second batch used the same frozen request bytes and received ten typed
replies with no errors: median 281.5 ms, maximum 407 ms. Neither batch retried
individual requests. Both batches and their receipts remain preserved.

| Measure | Second batch |
| --- | --- |
| Current digit load consistency | 50/50 classifications |
| Current individual opposition consistency | 10/10 classifications |
| Next-100-ms hold forecast, threshold 0.5 | 4/10, versus 6/10 for each simple comparator |
| Forecast Brier score | 0.28049, versus 0.4 for binary comparators and 0.25 for constant 0.5 |
| Angle choice | `hold_and_observe` in all ten cases |

One case repeats an identical input. The nine unique inputs give 4/9
threshold forecast accuracy. Latency exceeds the 100 ms horizon at real time.
Current consistency is largely arithmetic
already available locally. This small, curated and dependent sample does
not establish predictive improvement. No angle was physically executed, so
there is no physical angle-success score.

The local design uses Astra for the phase plan and Jev for asynchronous
bounded advice. `configs/isaac/astra-jev-press-plan-v2.json` explicitly allows
fresh local guards to advance the lever trajectory when model advice is
unavailable, preserves fresh model pauses for their original leases, and
latches accepted requests for Astra. Each fallback is logged as a local
decision. The original model-required mode remains the default. At this
checkpoint the new fallback mode has passed 256 focused tests but has not
yet completed an actual Isaac trial.

Reproduce the frozen live batch using
`scripts/dexterous/run_jev_contact_adjustment_study.py --study
out/local-jev/contact-adjustment-study-001 --output <fresh-directory>` with
`TYPESAFE_API_KEY` supplied only through the process environment. The key is
never part of the study or its repository artifacts.
