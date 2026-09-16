# OpenCode setup prompt for this endpoint

Paste the block below into a **freshly installed OpenCode**, running on whatever
model it starts with — a free OpenCode Zen model is enough. It does the install
itself: finds the `opencode.json` the operator sent, puts it in the global
config directory, verifies it and runs one real request against this node.

This is the companion to the file, not a replacement for it. quota-bot generates
both: `/opencode <name>` sends `opencode.json` and then this prompt, with the
provider and model ids filled in. Forward the two together.

The API key is **not** in the prompt — it is inside `opencode.json`, which is why
the prompt tells the agent never to print it or copy it anywhere tracked by git.

Why the prompt forbids "fixing" values: every number in the file is matched to a
measured limit of this deployment (`quota-bot/README.md` has the table). A model
that helpfully raises `limit.output` to the gateway's advertised 70,000, or drops
the timeouts back to OpenCode's defaults, produces failed requests — a 400 at
long context, or an abort during a cold prefill the node is still serving.

---

```text
Set me up to use my own model endpoint in opencode. Do the
work yourself, then tell me what to do next.

I have a file called opencode.json from the people who run
the endpoint. It is already tuned to their server.

1. Find it: look in ~/Downloads, ~/Desktop, /tmp and the
   current directory. If it is not there, stop and ask me
   to paste its contents.
2. Do not change any value inside it. Every number is
   matched to that server's limits, and "correcting" one
   causes failed requests.
3. It holds a live API key. Do not print it, do not copy
   it into a project directory or anything tracked by git,
   and do not pass it on a shell command line.
4. Install it globally, not per project:
   - mkdir -p ~/.config/opencode
   - if ~/.config/opencode/opencode.json already exists,
     copy it to opencode.json.bak first, then merge: keep
     my own settings, take every key from the new file,
     and tell me which ones collided.
   - otherwise move the file there.
   - chmod 600 ~/.config/opencode/opencode.json
5. Check it parses: jq . ~/.config/opencode/opencode.json
   (or python3 -m json.tool < that file).
6. Check opencode loaded it: opencode models qwen-gw
7. Test it for real:
   opencode run -m qwen-gw/MODEL "Reply with: OK"
   Allow up to 15 minutes and do not kill it early. The
   first token can take a couple of minutes when the
   server is busy; that is normal, not a hang. Report the
   exit status and the reply.
8. Then tell me to restart opencode and select
   qwen-gw/MODEL with /models. Change none of my other
   settings.

If a step fails, stop and show me the exact error instead
of working around it.
```

`MODEL` is replaced with the served model id (`MODEL_ID` in quota-bot's `.env`,
`qwen3.8-27b` today) when the bot renders this.
