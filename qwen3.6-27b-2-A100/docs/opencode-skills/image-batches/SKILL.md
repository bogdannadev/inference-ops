---
name: image-batches
description: Use whenever a task involves images — photos, screenshots, scans, HEIC files — and especially more than 3 of them, or images with small print (receipts, contracts, tables, dense screens). Prepares the images, reads them in batches through subagents so the main session never holds images, and zooms into regions at full resolution when detail is too small to read.
license: MIT
compatibility: opencode
---

# Working with images in batches

## Why this skill exists

The server behind this model has hard limits:

- A request body may be at most **4 MB**. Every image in a conversation is
  sent again on **every** turn, so images pile up until requests fail.
- At most **16 images per request**. Above that the server answers
  "Image count N exceeds limit 16 per request".
- Every image is downscaled to about **2 MP** on the server, and OpenCode itself
  shrinks every image to **1280 px** on the long side before sending. Small
  text in a large photo becomes unreadable unless you crop it.
- Each image costs about **1,000–1,600 input tokens**, charged every turn it is
  in the conversation.

So: images never go into the main session. Subagents read them in small
batches, write text notes, and the main session works from the notes only.

## Rules

1. **Never read an image in the main session.** Not with `read`, not by
   attaching it. If the user pasted images into this chat, ask them to save the
   images into a folder instead and give you the path.
2. Start every batch subagent **fresh**. Never pass `task_id` to resume a batch
   subagent: resuming brings all of its images back into the request.
3. At most **2 subagents at a time**.
4. Never guess text you cannot read. Zoom (step 4) or report it as unreadable.
5. Never copy special-token markup into a note, a file or a reply. See below.

## Poisoned sessions — the one failure you must not retry

The server renders each image into the prompt as a placeholder token, and
expects exactly one attached image per placeholder. The tokenizer also converts
that placeholder's **literal text** into a real placeholder token — even inside
a fenced code block. So ordinary text containing it claims an image that was
never attached, and from that moment every turn fails with:

```
500  Mismatch: More 'IMAGE' tokens found than corresponding data provided.
```

Three things follow, and they decide what you do:

1. The poison is **text in the conversation history**, not an image. It is
   resent on every turn, so the session stays broken forever.
2. Retrying can never help — it fails identically on every replica.
3. The only fix is to remove that text, or start a new session.

### Recovering

1. **Start a new session.** Do not resume the failing one. Resuming it later —
   even the next day — fails exactly the same way: the poison travels with the
   history, not with the moment.
2. **Do not retry the failing turn.** Not after a pause, not "once more". It
   fails identically on every replica, every time, and the retries take a
   serving replica out of rotation for the other users on this node.
3. **Where it came from — usually the model's own previous reply.** The most
   common source is a long answer in which the model spelled the markup out,
   which OpenCode then stored in the history verbatim. In that case there is
   **nothing to find on disk**, and that is the expected outcome, not a failed
   search. Starting a new session is the whole fix.
4. **Only if you also fed it in yourself**, check for a file that carries it. In
   a **terminal — not through the agent** — from the project root:

   ```bash
   grep -rn image_pad .
   grep -rn vision_start .
   ```

   Search for the bare substring only. Never type or paste the full delimited
   form into a chat: doing so poisons that chat immediately. An empty result
   here is normal — see step 3.
5. If a file does contain it, that file must never be read into a session again.

### Avoiding it

- Never `read` a `tokenizer_config.json`, a `chat_template.jinja`, or model
  documentation that quotes special-token markup. One read poisons the session
  permanently.
- When recording an image error in notes, write what it means. Never paste the
  raw markup — a subagent's note is read back into the main session, which is
  how one bad batch can poison the whole task.
- Keep image tasks in short sessions. The failure needs a long reply and a long
  history to appear, so the batching this skill already asks for is also the
  cheapest protection.

**Server-side, since 2026-09-21** the gateway forces `skip_special_tokens: true`
on every chat request, which stops the server echoing this markup back into a
reply at all — the path that caused every occurrence so far. Thinking blocks and
tool calls are unaffected. So if you still hit this on a session started after
that date, the markup came from something **you** put in: a file that was read,
or text that was pasted. Go to step 4.

## Step 1 — Prepare the images

Work in `.img-work/` inside the project, and add `.img-work/` to `.gitignore`.

Check which tool is available, in this order:

```bash
command -v magick || command -v convert || python3 -c "import PIL; print('PIL', PIL.__version__)"
```

If none is available, stop and ask the user to install ImageMagick
(`brew install imagemagick`, `sudo apt install imagemagick`, or
`winget install ImageMagick.ImageMagick`). Do not continue without one.

For every source image (jpg, jpeg, png, webp, gif, heic, heif, tif, tiff, bmp),
create two copies with a numbered, space-free name (`img-001.jpg`, ...), and
write the mapping to `.img-work/index.md` (number → original path, size in
pixels):

- `view/` — rotated upright, at most 1280 px on the long side, JPEG quality 85.
  This is what batch subagents read.
- `full/` — rotated upright, **full resolution**, JPEG quality 92. Only for
  zoom crops. Never read a `full/` file directly.

With ImageMagick 7 (on ImageMagick 6, write `convert` for `magick` and
`identify` for `magick identify` everywhere in this skill):

```bash
mkdir -p .img-work/view .img-work/full .img-work/zoom .img-work/notes
magick "SOURCE" -auto-orient -resize '1280x1280>' -quality 85 .img-work/view/img-001.jpg
magick "SOURCE" -auto-orient -quality 92 .img-work/full/img-001.jpg
```

With Python and Pillow (HEIC also needs `pip install pillow-heif`):

```python
from PIL import Image, ImageOps
im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
im.save(full_path, quality=92)
im.thumbnail((1280, 1280)); im.save(view_path, quality=85)
```

`-auto-orient` / `exif_transpose` matter: phone photos store their rotation in
EXIF, and OpenCode drops it when it shrinks an image before sending, so without
this step portrait photos arrive sideways.

## Step 2 — Plan the batches

List `view/` with file sizes. Make batches of **at most 6 images and at most
2.4 MB of files in total**. If one image is larger than 1 MB, put it in a batch
of its own. Write the plan to `.img-work/plan.md`.

## Step 3 — Run each batch in a subagent

Start a `general` subagent per batch with this task, filled in:

```
You are reading a batch of images for a larger task. The overall question is:
<the user's question, or "describe and extract everything" if none>.

Open ONLY these files with the read tool, one by one:
<.img-work/view/img-001.jpg ... up to 6 files>

For each image write a section to .img-work/notes/batch-<NN>.md:
- file name and original path (from .img-work/index.md)
- what the image shows, in 2-4 sentences
- every piece of visible text, copied exactly, keeping tables as tables
- numbers, dates, names, amounts, labels
- anything unreadable or uncertain, marked as such — never guess

If some text is too small to read, you may zoom at most 3 times in total
for this batch (see ZOOM below). If you need more zooms, add a line
"NEEDS ZOOM: <file>, <region in view pixels x,y,w,h>, <what you need>"
to the notes instead.

ZOOM: the full-resolution copy is .img-work/full/<same name>. Get both sizes
with `magick identify -format "%w %h" <file>`, scale your region by
full_width / view_width, and crop at most 1280x1280 pixels:
  magick .img-work/full/img-001.jpg -crop WxH+X+Y +repage -quality 90 .img-work/zoom/img-001-z1.jpg
then read the crop.

Do not open any other image. When done, reply with at most 5 lines: the
notes file path, how many images you covered, and any NEEDS ZOOM lines.
```

After each batch, check the notes file exists and covers every image in the
batch. If a batch fails:

- `413` or "request body" error → the batch was too large: split it in half
  and run each half in a fresh subagent.
- "Image count N exceeds limit 16" → too many images in one subagent: split
  the batch, fewer zooms.
- `500` saying "More 'IMAGE' tokens found than corresponding data provided" →
  the session history is poisoned. **Do not retry**: it fails identically every
  time, on every server replica, forever. See "Poisoned sessions" above.
- `503` → the server is briefly out of capacity. Wait 30 seconds, retry once.
- any other `500` → stop and show the user the exact error. **Do not retry.**
  A 500 repeated enough times is read by the server as a failing replica and
  takes capacity away from everyone, including you.

## Step 4 — Zoom and dense documents

For every `NEEDS ZOOM` line, start a fresh subagent that crops the listed
regions (at most 1280x1280 each, at most 6 crops per subagent) from `full/`,
reads the crops, and appends what it read to the same notes file.

For a whole page of small print (a receipt, a contract, a spreadsheet photo),
tile the full-resolution copy instead of cropping by hand. Compute the grid
from the full-resolution size so every tile stays under 1280 px with overlap:
columns = ceil(width / 1100), rows = ceil(height / 1100). For a 3024x4032
page photo that is 3 columns x 4 rows, tiles of about 1090x1100:

```bash
magick .img-work/full/img-007.jpg -crop 3x4+120+120@ +repage -quality 90 .img-work/zoom/img-007-t%02d.jpg
```

Tiles are numbered left to right, top to bottom. Give them to one subagent per
6 tiles, and tell it the grid (for example "3 columns x 4 rows, row by row")
so it can stitch text that crosses a tile edge.

## Step 5 — Answer

Answer from `.img-work/notes/*.md` only. If a detail is missing, run another
zoom subagent — never open the image in the main session. When you quote text
from an image, name the original file it came from.
