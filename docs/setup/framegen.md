# Frame Generation

chiaki-ng can generate extra frames between the frames sent by the console
(for example 60 fps stream → 120 fps on a 120 Hz display). Three backends:

| Backend | Platforms | GPU | Multiplier |
|---|---|---|---|
| NVIDIA DLSS Frame Generation | Windows (VC build) | GeForce RTX 40 series or newer; RTX 20 / RTX 30 with MFG Unlocker (through Streamline, see below) | 2x (3x / 4x on RTX 50 series; RTX 3070: 2x at 1080p, 4x at 720p) |
| AMD FSR 3.1 Frame Generation | Windows (VC build) | any GPU with Vulkan | 2x |
| Flow Frame Generation | Windows, Linux, Steam Deck | any GPU with Vulkan | 2x / 3x / 4x |

The DLSS and FSR runtimes (`nvngx_dlssg.dll`, `amd_fidelityfx_vk.dll`) ship
next to `chiaki.exe`; nothing extra has to be downloaded. Flow is chiaki-ng's
own interpolation and needs no runtime at all.

### Flow

Flow is built from libplacebo compute shaders, so it runs wherever the Vulkan
renderer runs. For each stream frame it

1. builds a luma pyramid (1/4 resolution and coarser),
2. estimates motion coarse to fine with symmetric block matching (seeded by
   the coarser level, neighbouring blocks and the previous frame's motion),
3. removes outliers and checks for scene cuts,
4. warps both stream frames along the motion to each in-between time and
   blends them, picking per pixel the vector that makes both frames agree.
   Where nothing agrees (areas being covered or revealed) it uses the
   temporally nearer frame; across a scene cut it repeats the nearer frame.

**Flow Quality** (Settings → Video): *Performance* analyses motion at 1/4
resolution (≈0.5 ms on an RTX 5070 Ti at 1440p), *Quality* at 1/2 resolution
(roughly 2–4x the analysis time; finer motion around small details). Use
Performance on handhelds such as the Steam Deck.

Static overlays (HUD, subtitles) are detected per pixel and kept in place,
including the anti-aliased edges and outlines of HUD text (a 1/2 resolution
mask of static pixels marks text areas);
next to them the scene's motion is looked up up to ~20 px away so the
background behind HUD text does not smear.

The overlay shows `Flow FG 2x` (`Flow FG HQ 2x` in Quality mode) and its GPU
time. Fast motion around object
edges can show some smearing in the covered/revealed areas.

## Requirements

- Renderer Backend: **Vulkan**
- Settings → Video → Frame Mapping: **Direct Mapping** (the default)
- An SDR stream (HDR streams are shown without frame generation)
- A display refresh rate of at least multiplier × stream fps
  (the multiplier is reduced automatically otherwise)

## Usage

Settings → Video → **Frame Generation**: pick DLSS, FSR or Flow, then the
**Frame Generation Multiplier** (DLSS and Flow). Works together with
[RTX Video Super Resolution](rtxvideo.md): frames are upscaled first, then
generated.

### NVIDIA VFG (Windows, experimental)

**NVIDIA VFG** uses the *Video Frame Generation* effect of the NVIDIA Video
Effects SDK: a model trained on video that only needs the two pictures, so it
has no motion-vector problem at all. It needs an NVIDIA RTX GPU (40/50 series;
30 series is not supported by NVIDIA) and is heavy: High costs tens of
milliseconds per frame pair at 1440p, which fits a 30 fps source (33 ms per
frame) better than 60 fps. Pick Low or Medium in **NVIDIA VFG Quality** if the
frame rate drops.

The NVIDIA runtime is not bundled with chiaki-ng; it is downloaded from NVIDIA
on request:

- **Installer:** tick *Download the NVIDIA VFG frame generation and RTX VSR
  runtime* (installs into `<chiaki folder>\vfx`).
- **Any time (also the portable zip):** Settings → Video (or Ctrl+O →
  Enhancement) → **NVIDIA Runtime** → *Download*, then restart chiaki-ng
  (installs into `%LOCALAPPDATA%\Chiaki\Chiaki\vfx`).
- **By hand:** run `install-nvidia-vfx.ps1` from the chiaki-ng folder
  (`-Destination <folder>` to choose where).

The script downloads NVIDIA's official `nvidia-vfx` 0.2.0.0 package from
`pypi.nvidia.com` (about 420 MB, checked against its SHA-256) and keeps only
`NVVideoEffects.dll`, `NVCVImage.dll`, `nvVFXVideoFrameGeneration.dll`,
`cudart64_12.dll`, `nvngx_vsr.dll` 1.8.3 and NVIDIA's license files (NVIDIA's
terms apply). When that `nvngx_vsr.dll` is newer than one next to `chiaki.exe`,
RTX VSR uses it too, which enables the Artifact Reduction and Streaming modes.

chiaki-ng looks for the runtime in `CHIAKI_VFX_PATH`, `<chiaki folder>\vfx`,
`%LOCALAPPDATA%\Chiaki\Chiaki\vfx`, `NV_VIDEO_EFFECTS_PATH` and
`C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects`, in that order. The
log shows `[framegen] VFG: Video Effects SDK x.y.z from ...` and
`NVIDIA VFG ready: WxH, quality ...`, or what is missing.

### DLSS motion vectors

DLSS Frame Generation is made for games, which hand it per-pixel motion
vectors and depth. A video stream has neither; with zero motion DLSS relies on
its own optical flow alone and drags the background along the edges of moving
characters ("jelly"). By default chiaki-ng now gives it per-pixel motion from
Flow's motion search, measured at the current frame: every pixel takes the
vector of the nearby block whose source in the previous frame matches it, so
edges stay sharp and HUD elements get zero motion. **DLSS Motion Vectors** in
Settings → Video (and Ctrl+O → Enhancement) switches between *Flow* and *Zero
motion* for comparison; **Flow Quality** sets the analysis resolution
(Performance ≈0.5 ms). The log shows `DLSS motion vectors: Flow`.

### DLSS Frame Generation through Streamline (RTX 20 / RTX 30)

RTX 20 and RTX 30 cards have no DLSS Frame Generation of their own; NGX
refuses it when chiaki-ng asks directly. [MFG Unlocker](https://github.com/dashdogy/MFG-Unlocker)
(MIT) turns it on for games that use NVIDIA Streamline, so chiaki-ng now has a
Streamline route as well: put MFG Unlocker's DLL next to `chiaki.exe` as
`version.dll` (the setup kit offers to download release 1.7.1 and checks its
SHA-256) and pick *NVIDIA DLSS Frame Generation* or *DLSS Frame Generation via
D3D12*. The Streamline 2.14.16 files (`sl.interposer.dll`, `sl.common.dll`,
`sl.dlss_g.dll`, `sl.reflex.dll`, `sl.pcl.dll`, MIT) come with chiaki-ng.

How it works: Streamline only generates frames on a swapchain it presents, so
chiaki-ng gives it a hidden window one row taller than the frame. The extra
row carries the frame number; generated frames come back with it marked, and
each presented frame is copied back into chiaki-ng's ring of shared buffers
together with a GPU timestamp of when it was ready.

- On an RTX 3070 the generated frames come one after the other, about 5 ms
  apart at 720p, 8.5 ms at 1080p and 24 ms at 2160p. 2x at 1080p is ready
  about 5 ms after the stream frame; 4x holds at 720p.
- When the frames are late (fewer than asked, or the last one ready later
  than the next stream frame), chiaki-ng lowers the multiplier one step, and
  raises it again when the timestamps say the next step fits, or tries it
  again after a while. The log says `Streamline: the GPU keeps up with N
  generated frames per stream frame at WxH (...)`.
- DLSS FG through Streamline runs while a chiaki-ng window has the focus
  (Streamline pauses it otherwise); the stream then shows the real frames.
- Route: `version.dll` next to `chiaki.exe` → Streamline; without it, NGX
  directly (RTX 40 / 50, AMD through NeuralHub). RTX 50 cards always use NGX
  (through Streamline their frames came only every 14-20 ms).
  `CHIAKI_DLSSG_ROUTE=sl` or `ngx` forces a route. Streamline's own log goes to
  `%TEMP%\chiaki-ng-streamline`.

### FSR 3.1 on Linux

Linux builds (AppImage, Flatpak, Steam Deck) build FSR 3.1 frame generation
from the AMD FidelityFX SDK 1.1.4 sources (`third-party/ffx-sdk`): the optical
flow and frame interpolation passes run directly in chiaki-ng's Vulkan
renderer, on any GPU, 2x. Windows builds use AMD's signed DLL. Both get
per-pixel motion from Flow's motion search: with zero motion FSR's own
optical flow did worse than a plain blend on video in tests.

### Generated frame clean-up

**Generated Frame Clean-up** (Settings → Video, or Ctrl+O → Enhancement)
post-processes the frames of every backend, after an idea of VeyAmux (the
Linux port of Veyra):

- **Protect HUD and text:** pixels with texture that stayed unchanged for 6
  real frames count as static (HUD, text, mini-map, logos). They get zero
  motion before DLSS/FSR run and the real pixel in the generated frames, and
  moving background next to them is taken from the frame where it is not
  covered, so text is not dragged along.
- **+ keep detail** (default): where the two frames, moved along the motion,
  do not agree (fine detail, foliage, the edge of a moving character), the
  smeared blend is replaced by the sharper of the two samples. Sharp real
  frames alternating with smeared generated ones is what reads as flicker
  while panning. Not applied to Flow, whose synthesis already does this.

Both use Flow's motion search (Flow Quality applies, about 0.5 ms at
Performance). FSR Frame Generation now also gets these motion vectors
instead of zero motion.

### Games under 60 fps (repeated pictures)

The console keeps sending 60 frames per second even when the game renders
fewer (30, 20, or a rate that moves between them), so some pictures arrive
again. Frame generation fed with a repeat would interpolate between two equal
pictures and then jump, so while it runs chiaki-ng drops every repeat and
generates frames only between real game pictures, each one spanning the
stream frames it stood for: the output frame rate stays what you picked
(2x of a 60 fps stream = 4x of 30 fps pictures = 120 fps).

A repeat is told from a new picture by the decoded picture itself
(`include/duplicateframe.h`): the luma plane, as 16 x 16 block means, is
compared with the last new picture; repeats change ten times or more less
than new pictures. A learned line between the two, the game's rhythm (every
2nd or 3rd frame new), the frame's size and a limit of 6 frames without a new
picture settle the close calls. A frame after a network loss, and a scene cut,
make frame generation start afresh instead of blending across them. Hardware
frames that are not copied to system memory, and `CHIAKI_FG_DEDUP=0`, fall back
to judging by the frame size alone. `chiaki-duplicateframetest` (not built by
default) runs the detector on synthetic 60, 45, 30, 20 fps and uneven scenes.

The stats overlay shows *stream fps* (with the game's rate, e.g.
`60 (game 30)`), *display fps* (e.g. `120 (30 real + 90 FG)`) and *latency*
(see Latency below). The log has
`Repeated pictures: judged from the decoded picture` and, every 10 s, a
`[dedup]` line with the pictures, repeats dropped and gaps.
Pick a multiplier whose output divides your refresh rate evenly (on 240 Hz:
2x or 4x of 60, not 3x).

## Latency

Frame generation has to wait for the next game picture before it can show the
frames in between, and that real picture comes after them: it adds
(multiplier − 1) / multiplier of a picture interval (60 fps: 2x ≈ 8 ms, 4x
≈ 12.5 ms; a 30 fps game twice that) plus its GPU time. DLSS 5 and DLSS
Super Resolution add their GPU time on top. Turn frame generation off for
latency sensitive games.

The stats overlay's *latency* line measures it: the time from the decoder
handing a frame over to the present that shows its real picture, averaged over
the last second with the maximum (e.g. `38 ms (max 45)`), Direct Mapping only.
The network, the PS5's encoder and decoding are not in it. Every 10 s the log
has a `[picture latency]` line (average, p50, p95, max, and the enhancements
that ran), to compare settings.

## Measuring artifacts

Frame generation can be measured on the real stream:

1. While streaming, **Ctrl+R** saves the video as the PS5 sends it (H.264/H.265, HDR as
   sent) to `Videos\chiaki-ng captures\chiaki-<time>[-hdr].h265`, up to 30 s from the
   next keyframe (chiaki-ng asks the PS5 for one); Ctrl+R again stops earlier. The stats
   overlay shows `REC 12 s` meanwhile. Pick a 60 fps game: its frames in between are the
   ground truth.
2. Turn it into PNG frames with ffmpeg:
   - SDR: `ffmpeg -f hevc -r 60 -i chiaki-….h265 -pix_fmt rgb24 frames\%05d.png`
   - HDR (PQ BT.2020 values as sent, 16-bit):
     `ffmpeg -f hevc -r 60 -i chiaki-…-hdr.h265 -vf scale=in_color_matrix=bt2020:in_range=tv:out_range=full -pix_fmt rgb48be frames\%05d.png`
3. `chiaki-fgdx12test --eval frames [--multiplier 2] [--cleanup 0|1|2] [--out worst]` feeds
   every 2nd frame (`--multiplier`) to frame generation as a real one and compares each
   generated frame with the real frame it stands for: luma PSNR / MAE, *art* (16×16 blocks
   off by more than 12/255) and *worse* (blocks clearly worse than showing the last real
   frame again), next to showing the last real frame again and a plain blend. `--out`
   keeps the worst frames as *truth | generated | difference* pictures.

DLSS Frame Generation through D3D12 gets zero motion vectors and finds the motion with
its own optical flow; `CHIAKI_DX12FG_FLOW_MOTION=1` gives it Flow's motion search instead
(an experiment: on RTX 30 through dlssg_for_sm86 it made no difference).

## Limitations

The stream carries no motion vectors or depth, so both backends get zero
motion vectors and a constant depth and rely on their own optical flow. Fast
motion and on-screen UI can show artifacts.

## If it crashes

Frame generation calls into the NVIDIA and AMD runtimes. A crash inside them is
caught where possible and only turns frame generation off for the session. If
chiaki-ng still closes while frame generation starts, it is switched off
automatically on the next launch (the log says so); turn it back on in the
settings once you want to retry.

The first frames log each step (`[framegen] frame N: ...`), which shows where
it stopped.

## Log messages

- `[framegen] ... initialized` and `Frame generation active: ... Nx` → working.
- `DLSS Frame Generation is not available on this GPU/driver` → RTX 40+ GPU and a current driver are needed (RTX 20 / RTX 30: MFG Unlocker as `version.dll`, see above).
- `Streamline: DLSS Frame Generation is not offered on this GPU` → MFG Unlocker is not next to `chiaki.exe`, or the driver is too old.
- `Could not load ...amd_fidelityfx_vk.dll` → the DLL is missing next to `chiaki.exe`.
- `Frame generation requires Direct Mapping` → switch Frame Mapping to Direct Mapping.
