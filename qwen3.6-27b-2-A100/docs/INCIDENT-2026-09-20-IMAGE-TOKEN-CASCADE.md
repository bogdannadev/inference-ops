# Incident 2026-09-20: one malformed image prompt took both replicas out

**2026-09-20 19:34:56–19:36:28 UTC. One consumer key, OpenCode desktop
2.0.10. 4 × 500 then 7 × 503 at the gateway. No other consumer was sending, so
user-visible impact was one person for 90 s.**

One conversation turn carried more image placeholder tokens than image
payloads. SGLang raised on it, returned **500**, and the router's default
retry × circuit-breaker settings turned that single bad request into a
fleet-wide 503. The engines were healthy throughout and never came under load.

The lasting damage was not the 90 s: r0's breaker was still `half_open` the
next morning, ~10 h later, with r1 serving 100% of traffic.

Companion docs: `OPERATIONS.md` (rolling a replica), `OBSERVABILITY.md`,
and the router block in `../docker-compose.yml` (policy and admission-control
rationale).

## 1. Timeline

All times UTC. Gateway rows from `gateway.requests`, engine rows from the
replica logs, breaker transitions from the router log.

| time | event |
|---|---|
| 19:13–19:31 | same session, body growing 397 KB → 497 KB, **all 200** |
| 19:34:56.560 | turn `a77b7a19…` finishes: 88,225 prompt / **11,202 completion** tokens |
| 19:34:56.727 | next turn starts — **167 ms later** — body 544,902 B → **500** |
| 19:34:59 | retry, identical 544,902 B → 500 |
| 19:35:00 | **r1 circuit breaker closed → open** |
| 19:35:05 | retry → 500 (now routed to r0) |
| 19:35:15 | retry → 500 |
| 19:35:16 | **r0 circuit breaker closed → open** |
| 19:35:25–19:36:28 | 7 × **503 `no_available_workers`**, engines never touched |
| 19:36:28 | client gives up |
| 19:36 → 23:56 | both breakers sit `open` — no traffic arrives to advance them |
| 2026-09-21 morning | r0 still `half_open`, 0 outcomes in 6 h; r1 served 81 |

The client resent the **identical 544,902-byte body** all 11 times.

## 2. Root cause: placeholders without payloads

A vision model does not receive "an image". It receives text containing image
placeholder tokens plus a **separate list of image payloads**, and the
processor walks the placeholders pulling one payload per placeholder, in order.

```
HEALTHY TURN  (19:31:44, 200)

  prompt:   ...text... [IMG] ...text... [IMG] ...text...
                         |                |
  payloads:            img1             img2            <- 2 asked, 2 given  OK


POISONED TURN (19:34:56, 500)

  prompt:   ...text... [IMG] ...text... [IMG] ...text... [IMG] ...
                         |                |                |
  payloads:            img1             img2              ???  <- 3 asked, 2 given
                                                           |
                                                           v
                                              iterator runs dry
                                              StopIteration
                                              RuntimeError  ->  HTTP 500
```

Engine log, on **both** replicas, 11 times each:

```
Mismatch: More 'IMAGE' tokens found than corresponding data provided.
  base_processor.py:1455  legacy_load_mm_data  -> StopIteration
  base_processor.py:1501  legacy_load_mm_data  -> RuntimeError
  (via qwen_vl.py _process_mm_data_uncached -> tokenizer_manager._tokenize_one_request)
```

`modalities` on the failing request shows **exactly 2 images supplied** —
unchanged from the turns that had just succeeded.

**This is NOT the 16-image cap.** `--limit-mm-data-per-request '{"image":16}'`
refuses cleanly with `Image count N exceeds limit 16 per request` *before any
image is decoded*. Different failure, and 2 is nowhere near 16.

**It is also not context overflow.** ~96.6k prompt tokens + 8,192 max output
against `--context-length 169000`. Not close.

### How an extra placeholder gets in — proven 2026-09-21

The tokenizer converts the placeholder's **literal text** into the real
placeholder token. Measured offline against the pinned image
(`AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")`, counting id 248056):

| IMAGE tokens | input text |
|---|---|
| 0 | plain prose, no markup |
| **1** | the delimited placeholder inside ordinary prose |
| **1** | the full vision markup inside ordinary prose |
| **1** | the same, inside a fenced code block |
| 0 | the same with spaces inserted between the delimiters |

So **any ordinary text carrying that markup claims an image nobody attached**,
and a code fence does not protect it. Three consequences:

1. The poison is **text in the history**, not an image. It is resent every turn,
   so the session is broken permanently — exactly the identical 544,902-byte
   body, 11 times.
2. Retrying can never succeed. Deterministic on every replica.
3. The only fix is to remove the text or start a new session.

Vision token ids for these weights: `vision_start` 248053, `vision_end` 248054,
`vision_pad` 248055, `image_pad` **248056**, `video_pad` 248057.

**⚠ Writing hazard.** Any file containing the delimited markup poisons a session
the moment an agent reads it — a tokenizer config, a chat template, model docs,
or a note quoting the error. That is why every diagram in this document writes
`[IMG]` and the literal form appears nowhere in this repo (verified by grep).
Search for the bare substring `image_pad`, never the delimited form, and do it
in a terminal rather than through an agent.

**No engine-level mitigation exists.** SGLang exposes no `split_special_tokens`
equivalent on `launch_server`, so user text can always inject control tokens.
The same vector applies to `<|im_start|>`-class tokens, which means user text
can break out of the chat template — a prompt-injection surface worth its own
assessment, not folded into this incident.

## 3. Why that turn and not the one before

The previous turn finished at 19:34:56.560 with an unusually long **11,202-token
reply**; the first 500 started at 19:34:56.727, **167 ms later**. The failing
request is therefore precisely the first one to carry that reply in its history,
and the body jumped 497,302 → 544,902 B (+47.6 KB) across that boundary.

The mechanism is proven (§2): placeholder markup in ordinary text becomes a real
IMAGE token. What is **not** proven is which message carried it in.

Note that `skip_special_tokens: True` strips genuine special tokens from model
output, so the model emitting the token *id* is ruled out. Two paths remain:

- **A file the agent read** — a tokenizer config, a chat template, model docs,
  or a subagent note quoting an error. One `read` poisons the session for good.
  This is the most likely path, and a `grep -rn image_pad` over the project
  finds it.
- **The model spelling the markup out character by character** in that
  11,202-token reply, which a long technical answer can do.

Bodies are not logged (`--log-requests-level 1`), so the specific message is
unrecoverable after the fact. The project-level grep is the decisive test, and
it is the first thing to run when a consumer reports this.

## 4. The amplifier: retries × circuit breaker

The request is **deterministic** — it fails identically on every replica, every
time. None of the following is set in `docker-compose.yml`; all are build
defaults (`--help` on the pinned image):

```
--retry-max-retries 5          (5 attempts per client request)
--cb-failure-threshold 10      (failures before a worker's breaker opens)
--cb-success-threshold 3       (successes to close from half-open)
--cb-timeout-duration-secs 60  (before half-open is attempted)
--cb-window-duration-secs 120
```

```
client request #1 (19:34:56)
   |- attempt 1 --> r1 --> 500      r1 failures: 1
   |- attempt 2 --> r1 --> 500                   2
   |- attempt 3 --> r1 --> 500                   3
   |- attempt 4 --> r1 --> 500                   4
   `- attempt 5 --> r1 --> 500                   5      client sees 500

client request #2 (19:34:59)   <- OpenCode retries the identical body
   `- attempt 1..5 -> r1 --> 500              6,7,8,9,10
                                                      |
                            threshold 10 -------------v
                                       19:35:00  r1 breaker OPENS

client request #3 (19:35:05)   <- r1 refuses; routing moves to r0
   `- attempt 1..5 -> r0 --> 500      r0 failures: 5

client request #4 (19:35:15)
   `- attempt 1..5 -> r0 --> 500                   10
                                       19:35:16  r0 breaker OPENS

client requests #5..#11 (19:35:25 -> 19:36:28)
   `- no worker left ------------------> 503 no_available_workers
                                         ~450 ms of pure retry backoff,
                                         engines never contacted
```

The governing arithmetic, with **no routing policy term in it**:

```
  cb_failure_threshold (10)  /  attempts per request (5)  =  2 requests kill a worker
  2 workers  x  2 requests                                =  4 requests kill the fleet
```

Observed: exactly 4 × 500, then 503s.

Counters that close the arithmetic:

| metric | value | meaning |
|---|---|---|
| `smg_http_responses_total{500}` | 4 | client-visible 500s |
| `smg_http_responses_total{503,no_available_workers}` | 7 | client-visible 503s |
| `smg_worker_retries_total` | 44 | 11 requests × 4 retries |
| `smg_worker_retries_exhausted_total` | 11 | every request exhausted them |
| engine 500s, r0 / r1 | 11 / 11 | 20 attempts + 2 half-open probes = 22 |

The 503 path spends ~406 ms of retry backoff (50 ms × 1.5, jitter 0.2) and
never reaches a worker — which is why every 503 row shows ~450 ms of upstream
time and the engines stayed idle.

## 5. The shape is policy-independent

The breakers opened 16 s apart, after requests 2 and 4 — so each request's five
attempts hammered a *single* worker, and routing only moved once that worker
tripped. Under `round_robin` the requests alternate (req1→A, req2→B, req3→A
opens A, req4→B opens B) and you get the identical 4 × 500 then 503, with the
two breaker events closer together. `power_of_two` on two idle workers is the
same alternation.

Policy history, for the record — `power_of_two` was retired six weeks before
multimodal traffic was viable here, so it never met this input:

```
2026-07-12  cache_aware
2026-07-28  power_of_two      (8d1081a)
2026-09-04  round_robin       (507c0e0)
2026-09-17  cache_aware       + --balance-abs-threshold 1
2026-09-19  image input fix goes live   <- images become usable at all
2026-09-20  this incident
```

Both observed occurrences of this cascade were under `cache_aware`, but both
were also after images became usable. **Two data points, two confounded
variables — do not attribute the cascade to the routing policy.**

The real variable is **fleet size**. With 20 workers, 40 bad requests would be
needed to walk the fleet and the client would have quit first. A two-replica
deployment makes a default circuit breaker extremely trigger-happy.

## 6. Aftermath: r0 stuck half-open

```
        10 failures / 120 s
  +----------+ ------------------> +----------+
  |  CLOSED  |                     |   OPEN   |
  | traffic  |                     | all 503  |
  |  flows   | <---------------+   +----+-----+
  +----------+   3 successes   |        | after 60 s
                               |        v
                       +-------+---------------+
                       |      HALF_OPEN        |
                       |  probes only          |
                       +-----------+-----------+
                                   | 1 failure
                                   `------> back to OPEN
```

Two things to understand about this machine:

1. **It advances on requests, not on a timer.** Both breakers sat `open` from
   19:36 to 23:56 purely because no traffic arrived all night. The 60 s timeout
   only says when half-open *may* be attempted.

2. **Half-open plus affinity routing is a deadlock.**

```
   r0: HALF_OPEN ---- needs 3 consecutive successes to close
         ^                     |
         |                     | cache_aware routes to the WARM replica
         |                     v
         +------------- r0 gets no traffic --> no successes --> stays HALF_OPEN
                                                                (self-reinforcing)
   r1: CLOSED --> serving 100%
```

Next morning: `smg_worker_cb_state{r0} = half_open`,
`cb_consecutive_successes{r0} = 0`, 0 outcomes in 6 h, r1 at 81 requests.

**The outage was policy-independent; this hangover is specifically a
`cache_aware` artifact.** It compounds the imbalance already documented in the
router block of `../docker-compose.yml`.

Gauge encoding: `0 = closed, 1 = open, 2 = half_open`.

## 7. Why a circuit breaker is the wrong tool here

```
  WHAT A BREAKER ASSUMES                WHAT WE HAD
  "host-correlated" failure             "request-correlated" failure

   req A --> r0   FAIL                   req X --> r0   FAIL
   req B --> r0   FAIL                   req X --> r1   FAIL
   req C --> r1   OK                     req Y --> r0   OK
   req D --> r1   OK                     req Y --> r1   OK

   r0 is sick.                           Nothing is sick.
   Routing around it WORKS.              Routing around it CANNOT WORK.
   Breaker = correct tool.               Breaker = deletes healthy capacity.
```

A breaker asks *"which machine is bad?"*. The bad thing was **the request**, so
the breaker walks the fleet, marks every healthy machine broken, and takes the
service down defending it.

Two retry layers multiply because neither knows the other exists:

```
  1 malformed prompt
    |- OpenCode resends it every ~10 s --------> 11 client requests
    `- router multiplies each by 5 attempts ---------> |
                                                       v
         first 4 requests -> 20 attempts reach engines -> 20 x 500
         last  7 requests -> breaker open, 0 engine load
                             (+2 half-open probes = 22 engine 500s)
```

This is a textbook **metastable failure**: a trigger pushes the system into a
degraded state, and a feedback loop (retry amplification) holds it there after
the trigger is gone. The still-half-open r0 *is* the system sitting in that
degraded equilibrium hours later.

The two rules broken, both from the AWS Builders' Library: **retry only when
the dependency is healthy**, and **retry at a single point in the stack**.

## 8. What to change

Ranked by leverage.

1. **Alert on `smg_worker_cb_state != 0`.** Nothing watches it today; that is
   the only reason a replica sat out overnight unnoticed. Zero risk, and it
   catches every future instance of this whatever the trigger.

2. ~~**Break the amplification.**~~ **APPLIED 2026-09-21 07:10 UTC** (commit
   `af8b24e`): `--retry-max-retries 1`, `--cb-failure-threshold 10` (pinned,
   unchanged in value), `--cb-success-threshold 1`. A deterministic 500 now
   costs 1 failure credit instead of 5, so it takes ten requests per worker
   rather than two, and 11 credits can never open both breakers. Applied by
   recreating the router alone (`up -d --no-deps`) in a window with 0 in-flight
   and 0 queued requests; both workers re-registered healthy, and the recreate
   also cleared r0's stuck `half_open`. Rationale is in the router block of
   `../docker-compose.yml`.

   Replaying this incident under the new settings: that client still sees 11 × 500,
   **but no 503, no fleet outage, and no stranded replica** — the second worker
   is never touched. It is blast-radius containment, not a fix.

   **Also fixed the client side:** `docs/opencode-skills/image-batches/SKILL.md`
   told the agent to "wait 30 seconds and retry once" on a 500. For this failure
   that is exactly wrong — it cannot succeed and it feeds the amplification. The
   skill now separates 503 (retry once) from 500 (never retry) and carries a
   "Poisoned sessions" section.

3. **File the 400-not-500 bug upstream** with the serving-path traceback. A
   malformed prompt is a client error. A 4xx is never retried and never reaches
   the breaker, so the same request would have produced one failed turn and
   nothing else. This is the only fix that also covers the next poison pill.

Recovering a stuck replica, cheapest first:

- **Pin three probes at it** with `x-smg-target-worker` (present in this build,
  alongside `x-smg-routing-key`). Three successes closes the breaker. No config
  change, no restart. Check whether Caddy/Higress forward the header, or send
  to the router directly.
- **Re-register the worker** — `GET /workers` → `DELETE /workers/<id>` →
  `POST /workers` — which resets breaker state. This is what
  `deploy/roll-replica.sh` already does. **Not** a router restart, which drops
  in-flight work on both replicas.

It may also self-heal: half-open does admit probes, and under real daytime load
`cache_aware` will eventually pick r0. Watch before acting.

## 9. How this was traced (and the trap in it)

**Three clocks on this node, and `docker logs --since/--until` parses a bare
timestamp as HOST LOCAL time (UTC+5).** Always append `Z`.

| source | printed clock |
|---|---|
| `qwen36-27b-r0` / `r1` (engines) | **UTC+5** |
| `qwen36-27b-router` (smg) | UTC |
| Higress access log, ClickHouse, Prometheus | UTC |

An engine line reading `[2026-09-21 00:35:05]` is 19:35:05 UTC the day before.
The two shifts cancel confusingly: filtering with a bare local-time string
returns engine lines whose printed times *look* like the UTC window you asked
for while actually being 5 h earlier. This hid the root cause for several passes
and made the router look like it had answered without logging.

**Anchor on identifiers instead of time.** The router echoes the gateway's
`x-request-id` (`--request-id-headers`), so:

```bash
docker logs qwen36-27b-router --since 2026-09-20T00:00:00Z 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep "<request_id>"
```

joins gateway rows to router lines exactly, with no time arithmetic. Engine rows
join on `gateway.requests.chat_id = engine.requests.rid`.

Useful queries:

```bash
# client-visible 5xx by consumer
docker exec qwen36-27b-langfuse-clickhouse clickhouse-client -q \
  "SELECT consumer, status, count() n, max(ts) FROM gateway.requests
   WHERE status >= 500 AND ts > now() - INTERVAL 7 DAY
   GROUP BY consumer, status ORDER BY n DESC"

# router-side truth: responses, retries, breaker state
docker exec qwen36-27b-router sh -c 'curl -s http://localhost:29000/metrics' \
  | grep -E "^smg_(http_responses|worker_retries|worker_cb)"
```

The Higress access log carries fields ClickHouse does not
(`upstream_service_time`, `bytes_received`, `ai_log`), at
`/var/log/proxy/access.log*` inside `higress-gateway-1`.

## 10. References

- **SGLang Model Gateway docs** (the router was renamed; hence the `smg` module
  name): <https://docs.sglang.io/docs/advanced_features/sgl_model_gateway>.
  Retryable status codes are **408, 429, 500, 502, 503, 504** — which is why a
  500 is retried five times.
  **Trap:** the docs page defaults do not match our pinned image. Docs say
  `cb-failure-threshold 5 / cb-success-threshold 2 / cb-timeout 30 /
  cb-window 60`; our `--help` says **10 / 3 / 60 / 120**. Same disagreement as
  `--api-key`. Trust `--help` on the pinned image.
- **Upstream error-classification stack**, which would *not* have saved us —
  it reclassifies 429/503 as backpressure, but genuine 5xx still counts as a
  breaker fault:
  [#39463](https://github.com/sgl-project/sglang/pull/39463),
  [#39464](https://github.com/sgl-project/sglang/pull/39464),
  [#39465](https://github.com/sgl-project/sglang/pull/39465).
- **Same error message, adjacent path** (offline processor output, Qwen3-VL):
  [sgl-project/sglang#16803](https://github.com/sgl-project/sglang/issues/16803).
  Ours is the serving path and is worth filing separately.
- **Metastable Failures in Distributed Systems**, Bronson et al., HotOS '21 —
  retry amplification was the sustaining loop in >50% of incidents studied:
  <https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s11-bronson.pdf>
- **Metastable Failures in the Wild**, USENIX ;login: —
  <https://www.usenix.org/publications/loginonline/metastable-failures-wild>
- **Timeouts, retries, and backoff with jitter**, AWS Builders' Library —
  <https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter>
- **Precedent in this repo:** the same 500-on-both-replicas → breaker → 503
  cascade with a different trigger (GPU OOM in image preprocess, 2026-09-18/19)
  is documented in the `--image-processor-backend pil` comment in
  `../docker-compose.yml`. **Any request that deterministically 500s on every
  replica produces this outage shape.**
