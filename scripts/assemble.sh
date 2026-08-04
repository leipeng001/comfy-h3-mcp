#!/usr/bin/env bash
# Assemble H3 clips into a frame-exact cut with continuous audio beds.
#
#   ./assemble.sh cuts.txt [output.mp4]
#
# cuts.txt is one cut per line:  <file>  <in_frame>  <duration_frames>
# Lines starting with # are ignored. Example:
#
#   # file                    in    len
#   shot_a_00001_.mp4         10    29
#   shot_b_00001_.mp4          0    24
#
# Durations are in FRAMES because seconds round independently per cut and drift
# your audio marks off the beat. Everything downstream derives from the running
# frame total, so the timeline is exact by construction.
#
# Audio: by default the source audio of each cut is discarded and a single bed
# is built from BED_SRC. Set BEDS to a "file:start_frame" list to lay several
# beds across act boundaries instead. A hard cut to silence beats any fade.

set -euo pipefail

CUTS_FILE="${1:?usage: assemble.sh cuts.txt [output.mp4]}"
OUT="${2:-assembled.mp4}"
FPS="${FPS:-24}"
WORK="${WORK:-./_assemble}"
CRF="${CRF:-16}"
LUFS="${LUFS:--16}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found" >&2; exit 1; }

mkdir -p "$WORK"
rm -f "$WORK"/cut_*.mp4 "$WORK"/list.txt

f2s () { awk -v f="$1" -v r="$FPS" 'BEGIN{printf "%.6f", f/r}'; }

# ---------------------------------------------------------------- picture
total=0
i=0
: > "$WORK/list.txt"
while read -r src inf len _rest; do
  [ -z "${src:-}" ] && continue
  case "$src" in \#*) continue ;; esac
  [ -f "$src" ] || { echo "no such file: $src" >&2; exit 1; }

  out=$(printf "%s/cut_%03d.mp4" "$WORK" "$i")
  # -ss before -i seeks fast; -t (not -to) keeps the duration relative.
  # Re-encoding is deliberate: -c copy snaps to keyframes and wrecks short cuts.
  ffmpeg -nostdin -y -loglevel error \
    -ss "$(f2s "$inf")" -t "$(f2s "$len")" -i "$src" \
    -an -c:v libx264 -crf "$CRF" -preset slow -pix_fmt yuv420p \
    -r "$FPS" -video_track_timescale $((FPS * 1000)) \
    "$out"

  got=$(ffprobe -v error -count_packets -select_streams v:0 \
        -show_entries stream=nb_read_packets -of csv=p=0 "$out")
  [ "$got" = "$len" ] || echo "  warn: cut $i wanted $len frames, got $got" >&2

  echo "file '$(basename "$out")'" >> "$WORK/list.txt"
  total=$((total + len))
  i=$((i + 1))
done < "$CUTS_FILE"

[ "$i" -gt 0 ] || { echo "no cuts parsed from $CUTS_FILE" >&2; exit 1; }

ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "$WORK/list.txt" \
  -c copy "$WORK/picture.mp4"

pic_frames=$(ffprobe -v error -count_packets -select_streams v:0 \
             -show_entries stream=nb_read_packets -of csv=p=0 "$WORK/picture.mp4")
echo "picture: $pic_frames frames ($(f2s "$pic_frames")s), expected $total"

DUR=$(f2s "$total")

# ---------------------------------------------------------------- audio bed
# One continuous bed by default. Looped to cover the full length, then padded
# to exactly DUR - short beds shift every later cue and truncate the picture.
BED_SRC="${BED_SRC:-}"
if [ -z "$BED_SRC" ]; then
  BED_SRC=$(awk '!/^#/ && NF {print $1; exit}' "$CUTS_FILE")
fi

ffmpeg -nostdin -y -loglevel error -stream_loop -1 -i "$BED_SRC" \
  -t "$DUR" -vn -af "apad=whole_dur=$DUR" -ar 48000 -ac 2 "$WORK/bed.wav"

# Two-pass loudnorm: measure, then apply in linear mode. One pass pumps
# audibly on flat material, and H3 output sits around -40 LUFS.
meas=$(ffmpeg -hide_banner -nostats -i "$WORK/bed.wav" \
       -af "loudnorm=I=$LUFS:TP=-1.5:LRA=11:print_format=json" \
       -f null - 2>&1 | sed -n '/^{/,/^}/p')
get () { echo "$meas" | grep "\"$1\"" | sed 's/.*: *"\([^"]*\)".*/\1/'; }

ffmpeg -nostdin -y -loglevel error -i "$WORK/bed.wav" -af \
  "loudnorm=I=$LUFS:TP=-1.5:LRA=11:measured_I=$(get input_i):measured_TP=$(get input_tp):measured_LRA=$(get input_lra):measured_thresh=$(get input_thresh):offset=$(get target_offset):linear=true:print_format=summary" \
  -ar 48000 -ac 2 "$WORK/bed_norm.wav" 2>/dev/null

# loudnorm resamples and can return a few ms short; pad again afterwards.
ffmpeg -nostdin -y -loglevel error -i "$WORK/bed_norm.wav" \
  -af "apad=whole_dur=$DUR" -t "$DUR" -ar 48000 -ac 2 "$WORK/mix.wav"

# ---------------------------------------------------------------- mux
# No -shortest: the picture is the master and must never be truncated by audio.
ffmpeg -nostdin -y -loglevel error -i "$WORK/picture.mp4" -i "$WORK/mix.wav" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k "$OUT"

final=$(ffprobe -v error -count_packets -select_streams v:0 \
        -show_entries stream=nb_read_packets -of csv=p=0 "$OUT")
echo "wrote $OUT — $final frames @ ${FPS}fps"
[ "$final" = "$total" ] || echo "  warn: expected $total frames, got $final" >&2
