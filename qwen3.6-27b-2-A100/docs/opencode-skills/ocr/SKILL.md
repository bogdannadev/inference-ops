---
name: ocr
description: Use whenever text has to be read out of images — scans, photographed documents, receipts, invoices, forms, tables, handwriting, screenshots of text — or the user asks for OCR, transcription or extraction from pictures or image-only PDFs. Sends the images to the low-effort @ocr subagent in small batches so transcription takes seconds instead of minutes, then works from its text.
license: MIT
compatibility: opencode
---

# OCR: transcribe with @ocr, think in the main session

## Why this skill exists

This model thinks before it answers, and how long it thinks is set per agent.
At the server's default effort a single turn reading 3–5 pages spent 30,000+
thinking tokens and 4–6 minutes before writing one word of the transcript.
Transcription needs eyes, not deliberation. The `@ocr` subagent in this
provider's config runs at **low** effort with a transcriber prompt: it reads
the images and returns the text verbatim, quickly and cheaply.

So the split is: **@ocr reads, the main session thinks.** Analysis, checking
totals, answering questions about the document — that happens afterwards, in
the main session, on plain text.

## Before you start

1. Check that `@ocr` exists: it is listed among the subagents the `task` tool
   offers. If it is missing, the config file is older than this skill. Tell
   the user to ask the operator for a fresh `opencode.json`, and meanwhile use
   a `general` subagent with the same task text below (slower, but it works).
2. If the `image-batches` skill is installed, load it too: its Step 1 (prepare
   the images) and its rules on poisoned sessions apply here unchanged.

## Rules

1. **Never read an image in the main session.** Images are resent on every
   turn of a conversation and the request body is capped at **16 MB**. If the
   user pasted images into the chat, ask them to save them to a folder and give
   you the path.
2. **One fresh @ocr subagent per batch.** Never pass `task_id` to resume one:
   resuming brings all of its images back into the request.
3. **At most 2 @ocr subagents at a time.** The node decodes a few requests at
   once for all its users; more parallel subagents only queue.
4. **Never correct or guess the text.** Transcription errors that look like
   typos may be what the document actually says. Unreadable is `[illegible]`.
5. **Never copy special-token markup** (control tokens written as text in
   angle brackets with vertical bars) into a note, a file or a reply — see
   "Poisoned sessions" in `image-batches`.

## Step 1 — Prepare

Work in `.img-work/` and add it to `.gitignore`. Convert every source to an
upright JPEG with a numbered, space-free name and write
`.img-work/index.md` (number → original path, size in pixels):

```bash
mkdir -p .img-work/view .img-work/full .img-work/zoom .img-work/notes
magick "SOURCE" -auto-orient -resize '2048x2048>' -quality 88 .img-work/view/img-001.jpg
magick "SOURCE" -auto-orient -quality 92 .img-work/full/img-001.jpg
```

(ImageMagick 6: `convert` instead of `magick`. Python/Pillow: see
`image-batches` Step 1.) The server reads each image at up to about 2 MP,
roughly 1450×1450 pixels' worth, so a 2048 px view is all the detail it can
use — small print needs the zoom in Step 4, not a bigger file.

For an image-only PDF, render pages first, one image per page:

```bash
magick -density 200 doc.pdf -quality 88 .img-work/view/page-%03d.jpg
```

(or `pdftoppm -r 200 -jpeg doc.pdf .img-work/view/page`). If `pdftotext`
returns real text for a PDF, it is not image-only: use the text and skip OCR.

## Step 2 — Batch

At most **6 images and at most 9 MB of files per batch** (base64 makes a
request a third larger than the files). An image over 4 MB goes alone. Write
the plan to `.img-work/plan.md`.

## Step 3 — Transcribe

For each batch, start `@ocr` with this task, filled in:

```
Transcribe these images for: <the user's goal, e.g. "extract the invoice
lines", or "full transcript">.

Read ONLY these files, one by one:
<.img-work/view/img-001.jpg ... up to 6 files>

Write the text of each image to .img-work/notes/batch-<NN>.md under a
heading with its file name and its original path from .img-work/index.md.
Keep the original language and line order. Tables as Markdown tables,
forms as "field: value". Mark unreadable text [illegible]. Where small
print is unreadable at this size, add a line
"NEEDS ZOOM: <file>, <region x,y,w,h in view pixels>, <what is there>".

Reply with at most 5 lines: the notes path, images covered, NEEDS ZOOM lines.
```

Check each notes file exists and covers every image of its batch. On errors:

- `413` / "request body" → batch too large: split it in half, fresh subagents.
- "Image count N exceeds limit 16" → split the batch.
- `400` "contains an image placeholder with no image attached" (or an old
  `500` "More 'IMAGE' tokens found") → the session is poisoned. **Do not
  retry**; follow "Poisoned sessions" in `image-batches`.
- `503` → wait 30 seconds, retry once.
- any other `500` → stop and show the user the exact error. Do not retry.

## Step 4 — Zoom for small print

For each `NEEDS ZOOM` line, crop the region from `full/` (scale the view
coordinates by full_width / view_width), at most 1400×1400 pixels per crop:

```bash
magick .img-work/full/img-001.jpg -crop WxH+X+Y +repage -quality 90 .img-work/zoom/img-001-z1.jpg
```

Give up to 6 crops to one fresh `@ocr` subagent and have it append to the
same notes file. For a whole dense page (receipt, statement, spreadsheet
photo), tile it instead: columns = ceil(width / 1300), rows = ceil(height /
1300), `magick full.jpg -crop CxR+100+100@ +repage tile-%02d.jpg`, and tell the
subagent the grid so it can join lines that cross a tile edge.

## Step 5 — Answer

Work from `.img-work/notes/*.md` only: totals, comparisons, translation,
summaries and answers happen here, in the main session, at its normal effort.
When you quote text, name the original file. If something is missing, run
another zoom batch — never open the image in the main session.
