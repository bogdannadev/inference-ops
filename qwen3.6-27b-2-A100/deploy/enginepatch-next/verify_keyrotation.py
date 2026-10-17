"""Offline check of the key-rotation patch in the pinned image, no GPU.

    docker run --rm --cpuset-cpus 48-55 --network none \
      -e PYTHONPATH=/opt/enginepatch -e SGLANG_API_KEY_ALT=alt-key \
      -v $PWD/deploy/enginepatch-next:/opt/enginepatch:ro \
      --entrypoint python3 <pinned image> /opt/enginepatch/verify_keyrotation.py

Runs the real, patched sglang.srt.utils.auth.decide_request_auth. Test keys
only; never pass a real key here.
"""

import os
import sys
from types import SimpleNamespace

import sglang.srt.utils.auth as auth
from sglang.srt.utils.auth import AuthLevel
import sitecustomize as sc

fails = 0


def check(label, ok):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def allowed(token, level=AuthLevel.NORMAL, path="/v1/chat/completions",
            api_key="new-key", admin_api_key=None, decide=None):
    decide = decide or auth.decide_request_auth
    return decide(method="POST", path=path,
                  authorization_header=None if token is None else f"Bearer {token}",
                  api_key=api_key, admin_api_key=admin_api_key,
                  auth_level=level).allowed


check("alternate key set for this run", os.environ.get("SGLANG_API_KEY_ALT") == "alt-key")
check("patch installed", getattr(auth, "_key_rotation", False)
      and hasattr(auth.decide_request_auth, "__wrapped__"))
check("current key accepted", allowed("new-key"))
check("alternate key accepted", allowed("alt-key"))
check("wrong key rejected", not allowed("other-key"))
check("no key rejected", not allowed(None))
check("health open without a key", allowed(None, path="/health"))
check("no --api-key: unchanged (open)", allowed(None, api_key=None))
check("admin-force: alternate key never accepted",
      not allowed("alt-key", AuthLevel.ADMIN_FORCE, admin_api_key="admin-key"))
check("admin-optional with an admin key: alternate key rejected",
      not allowed("alt-key", AuthLevel.ADMIN_OPTIONAL, admin_api_key="admin-key"))
check("admin-optional without an admin key: alternate key as the current one",
      allowed("alt-key", AuthLevel.ADMIN_OPTIONAL))

# Unset variable: the patch must leave the function untouched.
original = auth.decide_request_auth.__wrapped__
os.environ["SGLANG_API_KEY_ALT"] = ""
stand_in = SimpleNamespace(decide_request_auth=original)
sc._patch_auth(stand_in)
check("unset: function untouched", stand_in.decide_request_auth is original)
check("unset: alternate key rejected", not allowed("alt-key", decide=stand_in.decide_request_auth))

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
