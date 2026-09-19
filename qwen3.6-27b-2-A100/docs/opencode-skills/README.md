# OpenCode skills for consumers

Skills a consumer installs next to the `opencode.json` that quota-bot generates.
Each one is a single `SKILL.md`; OpenCode loads it on demand through its `skill`
tool, so an installed skill costs nothing until a task needs it.

| skill | use it when | why it exists |
|---|---|---|
| [image-batches](image-batches/SKILL.md) | a task involves images, especially more than 3, HEIC photos, or small print | OpenCode resends every image in the history on every turn, the gateway caps a request body at 4 MB, the engine allows 16 images per request and caps each at 2 MP. The skill keeps images out of the main session, reads them in subagent batches, and crops regions at full resolution when text is too small |

## Install (on the consumer's machine)

```bash
mkdir -p ~/.config/opencode/skills/image-batches
cp SKILL.md ~/.config/opencode/skills/image-batches/SKILL.md
```

Restart OpenCode, then check it is listed. Write the list to a file:
`opencode debug skill` output piped straight into another command gets cut off.

```bash
opencode debug skill > /tmp/opencode-skills.json && grep -c '"image-batches"' /tmp/opencode-skills.json
```

The skill needs ImageMagick (or Python with Pillow) on the consumer's machine
for EXIF rotation, HEIC conversion and crops. It asks for it if missing.

## Facts the skill relies on (OpenCode v1.18.31, checked in source)

- `read` returns an image as a data-URL attachment of the raw file bytes
  (`tool/read.ts`): JPEG, PNG, GIF and WebP only, so HEIC fails.
- Tool-result images and pasted images are both resized to
  `attachment.image` (`session/processor.ts`, `session/prompt.ts` →
  `image/image.ts`). quota-bot sets 1280×1280 and `MAX_BODY_BYTES / 8`, so a
  crop larger than 1280 px is shrunk again before it is sent.
- Neither OpenCode nor SGLang applies EXIF orientation, so the skill rotates
  images upright before anything reads them.
- A `task` subagent starts with an empty context unless it is resumed with
  `task_id` (`tool/task.txt`), which is what keeps batches independent.
