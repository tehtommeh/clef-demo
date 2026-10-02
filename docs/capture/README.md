# Regenerating the README media

These scripts drive the running frontend with headless Chromium (Playwright) and record what it
shows. The numbers in the media are therefore real results from your stack.

```bash
python3 -m venv .venv-capture && . .venv-capture/bin/activate
pip install playwright && playwright install chromium

make up                                              # the stack must be running
python docs/capture/screenshots.py docs/media shots  # full-page PNGs, one per tab
python docs/capture/screenshots.py /tmp/walk video   # the 50 s walkthrough (.webm)
python docs/capture/clips.py /tmp/clips              # one short .webm per feature
```

Convert the recordings with ffmpeg. The `.trim` file next to each clip holds the seconds of page
loading to cut from its start:

```bash
# walkthrough -> MP4
ffmpeg -i /tmp/walk/*.webm -vf "scale=1280:-2,fps=30" -c:v libx264 -crf 26 \
  -pix_fmt yuv420p -movflags +faststart docs/media/walkthrough.mp4

# clip -> GIF (GitHub plays GIFs inline in a README; it does not play committed .mp4 files)
n=router; ffmpeg -ss $(cat /tmp/clips/$n.trim) -i /tmp/clips/$n.webm \
  -vf "fps=10,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  docs/media/clip-$n.gif
```
