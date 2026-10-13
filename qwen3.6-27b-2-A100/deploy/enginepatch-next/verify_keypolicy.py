"""Offline check of patch 5 (keypolicy.py) in the pinned image, no GPU.

    docker run --rm --cpuset-cpus 48-55 --network none -e HF_HUB_OFFLINE=1 \
      -e PYTHONPATH=/opt/enginepatch -e KEYPOLICY_PATH=/tmp/kp/policy.json \
      -v $PWD/deploy/enginepatch-next:/opt/enginepatch:ro \
      -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
      --entrypoint python3 <pinned image> /opt/enginepatch/verify_keypolicy.py

1. Both hooks install on the real classes.
2. The real TokenizerManager._validate_one_request, patched, holds each key to
   its window and leaves the engine-wide checks alone.
3. The effort the template actually receives, computed the way v0.5.21's
   _process_messages + _apply_jinja_template merge it, rendered with the real
   chat template: the instruction tokens say which effort the model saw.
"""

import json
import os
import sys
import time

os.makedirs("/tmp/kp", exist_ok=True)
PATH = os.environ["KEYPOLICY_PATH"]
with open(PATH, "w") as f:
    json.dump({"version": 1, "default": {"context": 169000},
               "consumers": {"wide": {"context": 262144},
                             "ocr": {"context": 200000, "effort": "low"}}}, f)

import sglang.srt.entrypoints.openai.serving_chat as sc  # noqa: E402
import sglang.srt.managers.tokenizer_manager as tmm  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
import keypolicy as kp  # noqa: E402

fails = 0


def check(label, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")


check("chat hook installed", getattr(sc.OpenAIServingChat, "_keypolicy_effort", False))
check("tokenizer hook installed", getattr(tmm.TokenizerManager, "_keypolicy_context", False))

# --- 2. context -------------------------------------------------------------
# Upstream's validator reads a few launch flags from the runtime context, which
# only a launched server publishes. Stand in the defaults this deployment runs.
from types import SimpleNamespace  # noqa: E402

_features = SimpleNamespace(enable_encoder_swa_bounded_replay=False,
                            enable_custom_logit_processor=False,
                            return_hidden_states_mode=None)
tmm.get_exec = lambda: SimpleNamespace(features=_features)
if hasattr(tmm, "get_server_return_hidden_states_mode"):
    _req_mode = tmm.get_request_return_hidden_states_mode
    tmm.get_server_return_hidden_states_mode = lambda: _req_mode(False)
tm = object.__new__(tmm.TokenizerManager)
tm.context_len = 262144
tm.num_reserved_tokens = 0
tm.allow_auto_truncate = False
tm.validate_total_tokens = True
tm.is_generation = True


def req(consumer, n_input, max_new=None):
    obj = GenerateReqInput(text="x", sampling_params={"max_new_tokens": max_new},
                           custom_labels={"consumer": consumer} if consumer else None)
    try:
        tm._validate_one_request(obj, list(range(n_input)))
        return "ok", obj.sampling_params.get("max_new_tokens")
    except ValueError as e:
        return "400", str(e)


for consumer, n, mx, want in [
    (None, 150_000, None, "ok"),
    (None, 169_000, None, "400"),          # default window, as before the raise
    ("someone", 168_999, None, "ok"),
    ("someone", 160_000, 20_000, "400"),   # prompt + max_tokens over 169K
    ("wide", 250_000, None, "ok"),
    ("wide", 262_144, None, "400"),        # upstream's own check still fires
    ("ocr", 199_999, None, "ok"),
    ("ocr", 200_000, None, "400"),
]:
    got, detail = req(consumer, n, mx)
    check(f"context {consumer or '(no label)'} {n}+{mx}", got == want, f"{got} {str(detail)[:90]}")

got, mx = req("someone", 100_000)
check("unset max_tokens clamped to the key window", got == "ok" and mx == 69_000 - 1, str(mx))
got, mx = req("wide", 100_000)
check("full-window key keeps upstream's unset max_tokens", got == "ok" and mx is None, str(mx))

# A broken file keeps the last good copy.
with open(PATH, "w") as f:
    f.write("{not json")
time.sleep(1.1)
check("unparseable file keeps last good policy", req("wide", 250_000)[0] == "ok")
os.remove(PATH)
time.sleep(1.1)
check("missing file falls back to the default", req("wide", 250_000)[0] == "400")

# --- 3. effort ----------------------------------------------------------------
with open(PATH, "w") as f:
    json.dump({"version": 1, "default": {"context": 169000},
               "consumers": {"ocr": {"effort": "low"}, "quiet": {"effort": "none"}}}, f)
time.sleep(1.1)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B-FP8",
                                    revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a")
SERVER_DEFAULT = {"reasoning_effort": "xhigh", "preserve_thinking": True}


def rendered_tokens(body, consumer, patched):
    r = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **body)
    # serving_chat.py:1175, inside _convert_to_internal_request
    if r.chat_template_kwargs and "reasoning_effort" in r.chat_template_kwargs:
        r.reasoning_effort = r.chat_template_kwargs.pop("reasoning_effort")
    if patched:
        kp.resolve_effort(r, consumer)
    # serving_chat.py:1301, _process_messages
    ctk = dict(r.chat_template_kwargs or {})
    for k, v in SERVER_DEFAULT.items():
        ctk.setdefault(k, v)
    r.chat_template_kwargs = ctk
    # serving_chat.py:1655, _apply_jinja_template
    extra = {}
    if r.reasoning_effort is not None:
        extra["reasoning_effort"] = r.reasoning_effort
    extra.update(r.chat_template_kwargs)
    if r.reasoning_effort in ("none", "no_think"):
        extra["enable_thinking"] = False  # upstream's thinking-off path
    s = tok.apply_chat_template(r.messages and [{"role": "user", "content": "hi"}],
                                tokenize=False, add_generation_prompt=True, **extra)
    return len(tok.encode(s))


XHIGH, MEDIUM, LOW, OFF = 53, 11, 41, 13
for body, consumer, before, after in [
    ({}, None, XHIGH, XHIGH),
    ({"reasoning_effort": "low"}, None, XHIGH, LOW),
    ({"reasoning_effort": "medium"}, None, XHIGH, MEDIUM),
    ({"reasoning_effort": "high"}, None, XHIGH, XHIGH),
    ({"reasoning_effort": "minimal"}, None, XHIGH, LOW),
    ({"chat_template_kwargs": {"reasoning_effort": "low"}}, None, XHIGH, LOW),
    ({}, "ocr", XHIGH, LOW),                              # key default
    ({"reasoning_effort": "xhigh"}, "ocr", XHIGH, XHIGH),  # request wins
    ({}, "quiet", XHIGH, OFF),
    ({"reasoning_effort": "max"}, None, XHIGH, XHIGH),
]:
    if before is not None:
        b = rendered_tokens(body, consumer, patched=False)
        check(f"upstream {json.dumps(body)} {consumer}", b == before, f"{b} tokens")
    a = rendered_tokens(body, consumer, patched=True)
    check(f"patched  {json.dumps(body)} {consumer}", a == after, f"{a} tokens")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
