---
name: minimax-h3-prompting
description: >
  Write effective prompts for MiniMax-H3 video+audio generation, and assemble the
  resulting clips into a cut. Covers prompt structure, camera vocabulary, native
  audio prompting, reference-mode tagging, the model's hard constraints, and a
  frame-exact ffmpeg assembly recipe. Use this skill when generating video with
  MiniMax H3 / Hailuo (local ComfyUI or hosted), planning a multi-shot sequence,
  troubleshooting a clip that ignored the prompt, or editing generated clips
  together. Trigger for: MiniMax H3, Hailuo, h3_image_to_video,
  h3_reference_to_video, comfy-h3, text-to-video prompting, video shot list,
  cutting AI clips together, video prompt not working.
---

# Prompting MiniMax-H3

Claims below are marked by provenance:
**[Official]** ComfyUI/MiniMax docs · **[Community]** published guides ·
**[Measured]** verified on an RTX 4090 running this model locally.

---

## 1. Hard constraints — check these before writing anything

These are properties of the model. No prompt wording works around them.

| Constraint | Detail |
|---|---|
| **One continuous shot per generation** | **[Measured]** There is no shot-boundary mechanism. A prompt describing several shots yields one take that smears through them. |
| **Timecodes are noise** | **[Measured]** `[0s-1.5s] Shot 1:` etc. is tokenised as ordinary text. It does not schedule anything. |
| **Frame grid** | **[Official]** Length snaps *up* to `17k+5` frames at 24 fps. 124 ≈ 5.17s. Ask for 120, get 124. |
| **Duration** | **[Official]** Up to 15s per clip; trained range ≈ 124–362 frames. |
| **Canvas** | **[Official]** 768-px short edge, capped 768×1344, rounded to 32. |
| **Text renders as gibberish** | **[Measured]** On-screen words come out as letter-like shapes. Counters jitter (`00→29→24→22→26`) rather than climbing. Composite real text in post. |
| **Audio is quiet** | **[Measured]** Output sits near **−40 LUFS**. Normalise before delivery. |

### The consequence: plan a shot list, not a screenplay

For anything with cuts, **generate one clip per shot** and edit them together.
This is the single most important workflow point. Write each prompt as one
camera setup doing one thing.

---

## 2. Prompt formula

**[Community]** The widely-repeated ordering, and it matches observed behaviour:

```
[Camera shot + motion] + [Subject + description] + [Action]
  + [Scene + description] + [Lighting] + [Style/mood] + [Audio]
```

Element order matters — it signals what governs what.

**[Community]** Further rules that hold up:

- **Lead with a verb.** Dynamic verbs outperform noun-first openings.
- **Three movements maximum per shot.** More and they blur together.
- **Technical vocabulary over flowery adjectives.** These models are trained on
  professional cinematography; `anamorphic`, `shallow depth of field`,
  `rack focus`, `low-angle`, `bokeh`, `35mm` are levers. "Beautiful" is not.
- **Keep it under ~150 words.** Dense nouns and verbs, not a wall of text.
- **Say how things move**, not that they move: "drifting slowly left to right",
  "sharp whip pan", "rapid vertical ascent".

### Camera vocabulary that lands

`close-up` · `extreme close-up` · `medium shot` · `wide shot` · `low-angle` ·
`high-angle` · `Dutch angle` · `over-the-shoulder` · `POV`
`slow push-in` · `pull-back` · `dolly` · `tracking shot` · `crane shot` ·
`pan left/right` · `tilt up/down` · `handheld` · `static locked-off`

**[Community]** Prefer simple phrasing: *"slow tracking shot left, focusing on
the subject's face"* beats an elaborate camera-operator paragraph.

---

## 3. Audio — it is generated in the same pass

**[Official]** H3 produces native stereo audio jointly with the video, not as a
post-process. **Describe it in the same prompt**, as its own clause:

```
Audio: gusting wind, rapid footsteps on wet concrete, distant city ambience,
a low tense score underneath, a percussive accent hit on each leap.
```

**[Measured]** It genuinely responds — wind, footsteps, room tone, and score all
appear. Two caveats: output is very quiet (see §1), and **each clip carries its
own audio**, so a fast cut chatters. See §6.

---

## 4. Reference mode — assign every reference a job

**[Official]** Tag references in **connection order**, 1-based **per type**:
`<Picture 1>`, `<Video 1>`, `<Audio 1>`. Limits: **9 images, 3 videos, 3 audio**.

The rule that matters: **state what each reference is for** — identity, style,
motion, camera, or voice. Don't attach files and hope.

MiniMax's own demo prompt:

> "Reference the Hitchcock camera movement from `<Video 1>`, have the character
> in `<Picture 2>` sing, with the vocals matching `<Audio 3>`."

**This is how you get consistency across shots.** Generate shot 1, pull a clean
frame of the character or location, and pass it as `<Picture 1>` to every
subsequent shot with an explicit job: *"the man in `<Picture 1>`, same coat and
build, now seen from a low angle."*

`ref_image_size`: `match` is fast; `max` uses a 2048-px short edge for better
identity fidelity but is **several times slower**, since reference tokens ride
through every sampling step.

---

## 5. Resolution and iteration workflow

**[Official]** Full-quality 16:9 is **~1.0 MP (1344×768)**. The stock ComfyUI
template ships **864×480 (0.4 MP)** as its draft setting.

**[Measured]** on a 4090 at 124 frames, with sage attention active:

| Config | Time |
|---|---|
| 864×480, 20 steps | ~3m 45s |
| 1344×768, 30 steps | ~13m 15s |

**Workflow:** iterate prompts at 864×480, then re-run keepers at 1344×768 with
the **same seed**. Note the caveat — seed is not a guarantee of the same image
across a resolution change; expect a related composition, not an identical one.

**Sampler:** `res_multistep` / `simple` at **20 steps** is the template default.
It is higher-order, so 20 steps here is not a downgrade from 30 of `euler`.

---

## 6. Assembling clips into a cut

Fast cutting from 5-second takes is an **edit**, not a generation. Six things
that will bite, all learned the hard way:

1. **Specify durations in frames, not seconds.** At 24 fps, 0.9s is 21.6 frames.
   Each cut rounds independently and your timeline marks drift off the beat.
   Derive marks from the running frame total so they are exact by construction.
2. **Re-encode; never `-c copy` for the cuts.** `-ss` with stream copy snaps to
   keyframes and destroys frame accuracy — fatal at 1–2s durations.
3. **Use `-ss` before `-i` with `-t`** (not `-to`) to avoid relative-timestamp
   surprises.
4. **Never `-shortest` on the final mux.** Seeking into AAC loses a few ms to
   decoder priming and `loudnorm` returns short; either will silently clip
   frames off your picture. Pad audio to exact length instead.
5. **`amix` normalises by default.** `weights=1 0.9` divides everything by 1.9.
   Pass `normalize=0`.
6. **Two-pass `loudnorm`, linear mode.** Source is ~−40 LUFS; target −16 LUFS
   for web. One-pass pumps audibly on material this flat.

### Audio strategy

Do **not** carry thirteen audio fragments under thirteen picture cuts — it
chatters. Build **two or three continuous beds** spanning the act boundaries,
and let them change only where the story turns. **A hard cut to silence on the
key beat does more than any level automation.**

### Glob safety

Match source files **exactly**. `bsod_H_*.mp4` happily matches `bsod_H_mouse`
before `bsod_H_torch`, and re-renders leave `_00001_`/`_00002_` siblings.

A working reference implementation is in this repo at `scripts/assemble.sh`.

---

## 7. Quick checklist

- [ ] One camera setup per generation — cuts come from editing
- [ ] Leads with a verb; ≤3 movements; under ~150 words
- [ ] Technical camera/lens vocabulary, not adjectives
- [ ] Audio described as its own clause
- [ ] References tagged `<Picture 1>` etc., each with a stated job
- [ ] No on-screen text expected to be legible
- [ ] Drafting at 864×480; finals at 1344×768
- [ ] Length on the 17k+5 grid (124, 141, 158, …)
- [ ] Loudness normalised before delivery

---

## Sources

- [MiniMax H3 — ComfyUI docs](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [MiniMax H3 Day-0 Support in ComfyUI](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)
- [Hailuo Minimax Prompt Guide — Segmind](https://blog.segmind.com/hailuo-minimax-ai-video-prompt-guide/)
- [5 Tips for Minimax AI Video — Curious Refuge](https://curiousrefuge.com/blog/5-tips-for-using-minimax-ai-video-generator-by-hailuo-ai)
- [AI Video Prompt Guide — LTX](https://ltx.io/blog/ai-video-prompt-guide)

**[Measured]** items come from runs on this machine and supersede general
community advice where they conflict.
