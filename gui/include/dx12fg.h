#ifndef CHIAKI_GUI_DX12FG_H
#define CHIAKI_GUI_DX12FG_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

// DLSS Frame Generation through the NGX *D3D12* API, driven from a Vulkan renderer.
//
// chiaki-ng renders with Vulkan, but tools that provide DLSS Frame Generation on GPUs
// without NVIDIA's own runtime (NeuralHub / DLSS-NR on AMD) only answer NGX calls made
// through D3D12. This class bridges the two: it owns a D3D12 device on the same GPU,
// shares buffers and a fence with the Vulkan device, and runs the NGX D3D12 feature on
// them.
//
//  - Buffers are created by D3D12 (shared committed buffers) and imported into Vulkan
//    through VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT. Vulkan copies the real
//    frame and the motion vectors into them and copies the generated frames out of them.
//  - One D3D12 fence, shared and imported into Vulkan as a timeline semaphore
//    (VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT), orders the two sides.
//  - The NGX D3D12 runtime used by NeuralHub only opens after the first Present of a
//    swapchain on the device, so init() creates a hidden window with a swapchain and
//    presents once.
//
// Everything runs on the thread that calls it (no worker thread). It depends on nothing
// but Vulkan, D3D12, DXGI and NGX, so it can be tested without the rest of chiaki-ng.
class Dx12Fg
{
public:
    static constexpr int kMaxOutputs = 3;

    struct VulkanDevice
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    };

    // level: 0 info, 1 warning
    using LogFn = void (*)(int level, const char *message, void *user);

    Dx12Fg(const VulkanDevice &vk, LogFn log, void *user);
    ~Dx12Fg();
    Dx12Fg(const Dx12Fg &) = delete;
    Dx12Fg &operator=(const Dx12Fg &) = delete;

    // Creates the D3D12 device on the GPU of the Vulkan device, the hidden swapchain, the
    // NGX runtime and the shared fence; logs why it cannot.
    bool init();
    // Generated frames per real frame the runtime supports (at most kMaxOutputs).
    int maxGenerated() const;

    // (Re)creates the shared buffers, the D3D12 textures and the NGX feature for this
    // size and runs two warm-up evaluations (the runtime builds its pipelines then,
    // about 0.4 s). No GPU work may be pending on either side.
    bool ensureSize(int w, int h);
    bool sizeMatches(int w, int h) const;
    // Drops the size dependent state. No GPU work may be pending.
    void invalidate();

    // Timeline semaphore shared with D3D12 and the last value that has been (or will be)
    // signalled on it.
    VkSemaphore semaphore() const;
    uint64_t lastValue() const;
    void setLastValue(uint64_t value);

    // Record copies between a Vulkan image (VK_IMAGE_LAYOUT_GENERAL) and the shared
    // buffers. The color image is RGBA8 (VK_FORMAT_R8G8B8A8_UNORM), the motion image
    // VK_FORMAT_R16G16_SFLOAT, both w x h as given to ensureSize().
    void recordColor(VkCommandBuffer cmd, VkImage image);
    void recordMotion(VkCommandBuffer cmd, VkImage image);
    void recordDownload(VkCommandBuffer cmd, VkImage image, int index);

    // The D3D12 half of one real frame: waits (on the GPU) for `wait_value`, copies the
    // shared color buffer (and the motion buffer if `use_motion`, otherwise the motion
    // vectors stay zero) into the D3D12 textures, generates `count` frames between the
    // previous real frame and this one, copies them to the shared output buffers and
    // signals `signal_value`. `signal_value` is always signalled, also after a failure
    // (which is reported through the return value), so the Vulkan work waiting for it
    // never hangs. mvec_scale is applied to the motion vectors (NGX takes them
    // normalised). NeuralHub's runtime ignores the motion vectors (it finds the motion
    // itself), NVIDIA's own runtime uses them.
    bool generate(int count, uint64_t wait_value, uint64_t signal_value, bool reset, bool use_motion,
                  float mvec_scale_x, float mvec_scale_y);

    // The runtime call in progress, named in the log if the runtime crashes.
    const char *stage() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_DX12FG_H
