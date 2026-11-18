Fork of chiaki-ng 1.10.0 with NVIDIA RTX Video Super Resolution, Frame Generation (NVIDIA VFG / DLSS / FSR / Flow) and Windows streaming tweaks.

## Downloads
| File | Platform |
|---|---|
| `chiaki-ng-*-win_x64-setup.exe` | Windows x64 installer |
| `chiaki-ng-*-win_x64-portable.zip` | Windows x64 portable |
| `chiaki-ng-*-linux_x86_64.AppImage` | Linux x86_64 (`chmod +x`, then run) |
| `SHA256SUMS.txt` | checksums |

RTX Video Super Resolution, NVIDIA VFG and DLSS Frame Generation are **Windows only**. **FSR 3.1 and Flow Frame Generation work on Windows and Linux** (any GPU, incl. Steam Deck). The Linux build also has the general fixes, the FPS stats and the disconnect dialog changes.

## New in 1.10.0-rtx.4
- **NVIDIA VFG frame generation** (RTX 40/50, Windows), 2x–4x: NVIDIA's video-trained interpolation model (Video Effects SDK), the least "jelly" option for 30 fps games. Quality Low / Medium / High. The SDK calls run on a worker thread; own scene-cut test instead of the SDK's.
  - **NVIDIA runtime download**: Settings → Video (or Ctrl+O → Enhancement) → *NVIDIA Runtime* → Download, or the installer's *Download the NVIDIA VFG … runtime* option. Fetches NVIDIA's official `nvidia-vfx` package (~420 MB, SHA-256 checked) and installs the VFG libraries plus `nvngx_vsr.dll` 1.8.3 (all RTX VSR modes). Not bundled; NVIDIA's license terms apply.
- **Generated Frame Clean-up** (all backends): HUD and text stay sharp and are not dragged along; optional detail pass against smeared in-between frames.
- **DLSS / FSR motion vectors** from chiaki-ng's own motion search (DLSS: Zero / Flow / Flow reversed).
- **FSR 3.1 Frame Generation on Linux**, built from the AMD FidelityFX SDK 1.1.4 sources.
- **30 fps games in a 60 fps stream**: repeated pictures are detected and dropped while frame generation is on.
- **Stats overlay**: styles Mini / Compact / Default / Full and nine positions (e.g. one line at the top center); it no longer stays above other applications.
- RCAS sharpening now also applies with RTX VSR or frame generation active.

## Features
- **RTX Video Super Resolution** (NVIDIA RTX 20+, Vulkan renderer): AI upscaling of the stream to the window size, modes Low / Medium / High / Ultra.
  Needs `nvngx_vsr.dll` (NVIDIA RTX Video SDK) next to `chiaki.exe` – not bundled. Artifact Reduction / Streaming modes need `nvngx_vsr.dll` 1.8.2+ and are greyed out with older DLLs.
- **Frame Generation**, backend chosen in Settings:
  - **DLSS Frame Generation** (NVIDIA RTX 40/50), 2x–4x
  - **NVIDIA VFG** (NVIDIA RTX 40/50, Windows), 2x–4x – runtime downloaded on request
  - **FSR 3.1 Frame Generation** (any Vulkan GPU, Windows + Linux), 2x
  - **Flow Frame Generation** (any Vulkan GPU, Windows + Linux), 2x–4x – chiaki-ng's own optical flow interpolation, no extra runtime

  Needs Frame Mapping = Direct Mapping and a display running at least twice the stream frame rate (e.g. 120 Hz for 60 fps).
- **Enhancement panel (Ctrl+O → Enhancement)**: RTX VSR mode, frame generation backend / multiplier / Flow quality, RCAS strength and border fill, changed live without leaving the stream.
- **Stats overlay**: stream fps / display fps, RTX VSR / Frame Gen status and GPU time.
- **Disconnect dialog**: Cancel button and × to keep streaming.

## Fixes and tuning
- Packet loss statistic fixed when sequence numbers wrap around.
- Windows: qWAVE QoS on the stream sockets, MMCSS "Games" priority for the network threads, 1 ms timer resolution, no power throttling while streaming.
- From upstream: use-after-free crashes on session end / controller disconnect / exit, Takion send buffer used after the receive loop ended, reorder queue not clearing dropped entries.
- Frame Gen crash guard: if the app crashes while Frame Gen starts, Frame Gen is switched off on the next launch.

## Notes
- HDR streams skip Frame Gen.
- Bundled runtimes: `nvngx_dlssg.dll` (NVIDIA DLSS SDK), `amd_fidelityfx_vk.dll` (AMD FidelityFX SDK, MIT); licenses in the `licenses` folder. The NVIDIA VFG runtime is not bundled (see above).
- Guides: [RTX Video](https://github.com/FzeNiX/chiaki-ng/blob/main/docs/setup/rtxvideo.md), [Frame Generation](https://github.com/FzeNiX/chiaki-ng/blob/main/docs/setup/framegen.md).
- Pre-release: if something breaks, attach the session log (lines starting with `[framegen]` / `[rtx-vsr]`).
