// chiaki-ng: frame interpolation runs into chiaki-ng's own textures, so the
// SDK's swapchain replacement (not compiled) is never used; ffx_vk.cpp still
// references this entry point of it.
#include <FidelityFX/host/backends/vk/ffx_vk.h>

FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const *)
{
    return FFX_ERROR_INVALID_ARGUMENT;
}
