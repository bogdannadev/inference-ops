# OpenCode skills for consumers

Skills a consumer installs next to the `opencode.json` that quota-bot generates.
Each one is a single `SKILL.md`; OpenCode loads it on demand through its `skill`
tool, so an installed skill costs nothing until a task needs it.

| skill | use it when | why it exists |
|---|---|---|
| [image-batches](image-batches/SKILL.md) | a task involves images, especially more than 3, HEIC photos, or small print | OpenCode resends every image in the history on every turn, the gateway caps a request body at 4 MB, the engine allows 16 images per request and caps each at 2 MP. The skill keeps images out of the main session, reads them in subagent batches, and crops regions at full resolution when text is too small |

## Install in the OpenCode desktop app (copy from GitHub)

The skill is one text file. OpenCode reads skills when it starts, from
`~/.config/opencode/skills/<name>/SKILL.md` (Windows:
`%USERPROFILE%\.config\opencode\skills\<name>\SKILL.md`).

1. Open the raw file and copy all of it:
   https://raw.githubusercontent.com/bogdannadev/inference-ops/master/qwen3.6-27b-2-A100/docs/opencode-skills/image-batches/SKILL.md
2. In the OpenCode app, start a new session and send this message, then paste
   the copied text under it:

   ```
   Save the text below as ~/.config/opencode/skills/image-batches/SKILL.md,
   creating the folders if needed. Copy it exactly, change nothing.
   ```

   OpenCode asks for permission to write outside the project; allow it.
   Or save the file yourself in a text editor under that path. The name must
   be exactly `SKILL.md`, and the file must start with the `---` block.
3. Quit and reopen the app.
4. In a new session, ask "Which skills do you have?". The answer should list
   `image-batches`.

## Install from a terminal

```bash
mkdir -p ~/.config/opencode/skills/image-batches
cp SKILL.md ~/.config/opencode/skills/image-batches/SKILL.md
```

Restart OpenCode, then check it is listed. Write the list to a file:
`opencode debug skill` output piped straight into another command gets cut off.

```bash
opencode debug skill > /tmp/opencode-skills.json && grep -c '"image-batches"' /tmp/opencode-skills.json
```

`~/.agents/skills/image-batches/SKILL.md` works too: OpenCode also scans
`~/.agents/skills` and `~/.claude/skills` (`skill/index.ts`).

The skill needs ImageMagick (or Python with Pillow) on the consumer's machine
for EXIF rotation, HEIC conversion and crops. It asks for it if missing.

## Facts the skill relies on (OpenCode v1.18.31, checked in source)

- `read` returns an image as a data-URL attachment of the raw file bytes
  (`tool/read.ts`): JPEG, PNG, GIF and WebP only, so HEIC fails.
- Tool-result images and pasted images are both resized to
  `attachment.image` (`session/processor.ts`, `session/prompt.ts` →
  `image/image.ts`). quota-bot sets 1280×1280 and `MAX_BODY_BYTES / 8`, so a
  crop larger than 1280 px is shrunk again before it is sent.
- OpenCode resizes with photon (`image/image.ts`): decode, resize, re-encode
  as JPEG or PNG, which drops EXIF orientation. SGLang v0.5.20 does apply
  EXIF (`smart_to_rgb` → `ImageOps.exif_transpose`), but only an image small
  enough to pass through OpenCode unchanged still carries it. So the skill
  rotates images upright before anything reads them.
- Server limits the skill is sized for (unchanged by the 2026-09-19 engine
  work): 4 MB request body at the gateway, 16 images per request, 2 MP cap per
  image, `--max-running-requests 4` per replica (hence at most 2 subagents).
  A batch that still overflows the body gets a 413, which the skill handles by
  splitting the batch.
- A `task` subagent starts with an empty context unless it is resumed with
  `task_id` (`tool/task.txt`), which is what keeps batches independent.
