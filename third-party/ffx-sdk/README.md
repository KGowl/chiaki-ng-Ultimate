# AMD FidelityFX SDK 1.1.4 subset (MIT)

Used by chiaki-ng's Linux build for AMD FSR 3.1 frame generation
(`gui/src/ffxfi.cpp`): AMD ships no FSR runtime for Linux, so the optical
flow and frame interpolation components and the Vulkan backend are built from
source and driven into chiaki-ng's own textures (no swapchain replacement).
Windows builds keep using AMD's signed `amd_fidelityfx_vk.dll`.

- `include/`, `src/`, `LICENSE.txt`: from
  https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (tag v1.1.4),
  only the shared code, optical flow, frame interpolation and the Vulkan
  backend, MIT licensed by AMD. Taken in the Linux-ready form prepared by
  VeyAmux (https://github.com/8BiTw0LF/VeyAmux, `third_party/fidelityfx`),
  whose changes are:
  - `FFX_SDK_DEFAULT_CONTEXT_SIZE` doubled (`wchar_t` is 4 bytes on Linux),
  - the Vulkan backend's scratch tables 64-byte aligned (GCC emits aligned
    SSE stores), host-visible device-local memory accepted as a fallback,
  - a generic optical flow reduction for subgroups below 32 lanes,
  - `FFX_MAX_QUEUED_FRAMES` raised from 4 to 8.
- chiaki-ng change: `ffx_vk.cpp` falls back to the core
  `vkGetBufferMemoryRequirements2` when the KHR alias is not loaded.
- `chiaki/ffx_gcc_compat.h`: force-included MSVC compatibility definitions
  (`_countof`, `wcscpy_s`, ...). `chiaki/ffx_stubs.cpp`: the swapchain entry
  point the Vulkan backend references.
- `generated/`: the SPIR-V shader permutation headers, generated with
  `scripts/ffx-gen-shaders.py third-party/ffx-sdk third-party/ffx-sdk/generated`
  (glslangValidator) instead of the SDK's Windows-only FidelityFX_SC.exe.
  Rerun it after changing the shaders.
