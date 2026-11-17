#ifndef CHIAKI_GUI_FFXFI_H
#define CHIAKI_GUI_FFXFI_H

#include <vulkan/vulkan.h>

#include <memory>

// AMD FSR 3.1 frame generation built from the FidelityFX SDK 1.1.4 sources
// (third-party/ffx-sdk, MIT): the optical flow and frame interpolation
// components driven directly into the caller's command buffer, without the
// SDK's swapchain replacement or the prebuilt amd_fidelityfx_vk.dll. This is
// how FSR frame generation runs on Linux, where AMD ships no runtime; the
// approach follows VeyAmux (Linux port of Veyra).
//
// All images stay in VK_IMAGE_LAYOUT_GENERAL.
class FfxFiRunner
{
public:
    struct Image
    {
        VkImage image = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        int w = 0;
        int h = 0;
    };

    FfxFiRunner(VkDevice device, VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr get_device_proc);
    ~FfxFiRunner();
    FfxFiRunner(const FfxFiRunner &) = delete;
    FfxFiRunner &operator=(const FfxFiRunner &) = delete;

    // (Re)creates the contexts for this size. No work may be pending.
    bool ensure(int w, int h);
    // Records the frame half-way between the previous `color` and this one
    // into `output` (RGBA8). `motion`: per-pixel motion in pixels ("this
    // content was at p + mv in the previous frame"), rg16f; `depth`: r32f.
    bool record(VkCommandBuffer cmd, const Image &color, const Image &output, const Image &motion,
                const Image &depth, bool reset, float frame_time_ms);
    // Drops the contexts. No work may be pending.
    void invalidate();
    const char *stage() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_FFXFI_H
