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
    // Generated frames per real frame the runtime supports (at most kMaxOutputs). With Streamline,
    // as many as the GPU keeps up with at the stream's frame rate (measured, it changes).
    int maxGenerated() const;
    // The stream's frame interval: the time Streamline has for the generated frames of each.
    void setFrameInterval(float ms);

    // (Re)creates the shared buffers, the D3D12 textures and the NGX feature for this
    // size and runs two warm-up evaluations (the runtime builds its pipelines then,
    // about 0.4 s). No GPU work may be pending on either side.
    // `hdr`: the color buffers are RGBA16F (DXGI_FORMAT_R16G16B16A16_FLOAT /
    // VK_FORMAT_R16G16B16A16_SFLOAT) and DLSS-G is told they are HDR (colorBuffersHDR); they
    // hold the output's own values (BT.2020 PQ), not linear light. Otherwise RGBA8.
    bool ensureSize(int w, int h, bool hdr = false);
    bool sizeMatches(int w, int h, bool hdr = false) const;
    // Whether the last failed ensureSize() failed in the runtime (it refused to create the feature
    // or to evaluate it), not for want of memory or because a GPU wait gave up. An HDR size the
    // runtime refuses while the same size in SDR works means it does not take RGBA16F.
    bool sizeRefused() const;
    // Drops the size dependent state. No GPU work may be pending. When the D3D12 GPU work does not
    // finish (a wait that gave up) the buffers and the feature are left as they are (leaked), not
    // freed under the GPU; after a crash nothing is done (the runtime is not called again).
    void invalidate();

    // Timeline semaphore shared with D3D12 and the last value that has been (or will be)
    // signalled on it.
    VkSemaphore semaphore() const;
    uint64_t lastValue() const;
    void setLastValue(uint64_t value);
    // The value the GPU has reached on that timeline (never waits). Vulkan work that waits
    // for a higher value would hold up everything queued after it on the same queue.
    uint64_t completedValue() const;

    // Record copies between a Vulkan image (VK_IMAGE_LAYOUT_GENERAL) and the shared
    // buffers. The color image is VK_FORMAT_R8G8B8A8_UNORM, or VK_FORMAT_R16G16B16A16_SFLOAT
    // when ensureSize() was given `hdr`; the motion image VK_FORMAT_R16G16_SFLOAT; all w x h
    // as given to ensureSize().
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
    // Average GPU time of the frame generation evaluations of one generate() call in milliseconds
    // (0 if unknown; NGX route only).
    float fgGpuTimeMs() const;

    // DLSS Frame Generation through NVIDIA Streamline (NVIDIA, with MFG Unlocker's version.dll
    // next to chiaki.exe, or CHIAKI_DLSSG_ROUTE=sl; MFG Unlocker opens DLSS-G on RTX 20 / RTX 30
    // for Streamline only). generate() hands the frame to Streamline, which presents the
    // generated frames on a hidden swapchain over the following frame interval; each is copied
    // back as it is presented. They are picked up with framesReady() / takeFrames() before
    // recordDownload(). With NGX directly, the frames are there once `signal_value` is reached.
    bool asyncFrames() const;
    // Whether the generated frames of the last generate() are in, or are not coming (Streamline
    // presented that real frame without them, or they took too long). Never waits.
    bool framesReady();
    // How many generated frames of the last generate() came in (0..count), in order;
    // recordDownload(i) copies the i-th. Once per generate(), after framesReady().
    int takeFrames();

    // What the NGX D3D12 runtime offers; init() succeeds with either.
    bool fgAvailable() const;
    bool nrAvailable() const;

    // ---- neural rendering (DLSS 5) ----
    // NVIDIA: NGX feature 18 from nvngx_dlssnr.dll next to chiaki.exe, called through
    // nvngx.dll_dlssnr.dll (nrgate.cpp); upscaling is NVIDIA's DLSS Super Resolution
    // (nvngx_dlss.dll) after it, and HDR goes through an SDR proxy (the model is SDR only)
    // whose change is put back on the linear frame. Other GPUs: one DLSS Super Resolution
    // feature, which NeuralHub serves with its DLSS 5 neural rendering pass in front of the
    // upscaler (DLAA when input = output size; IsHDR for HDR). It has its own size,
    // textures and shared buffers next to the frame generation ones and uses the same
    // shared timeline semaphore (lastValue / setLastValue).
    struct NrConfig
    {
        int width = 0, height = 0;         // input (stream frame) size
        int out_width = 0, out_height = 0; // output size: the input size, or larger to upscale
        // RGBA16F linear light, 1.0 = SDR white (203 nits), BT.709 primaries; else RGBA8 sRGB.
        bool hdr = false;
        // NVIDIA feature 18 model: 0 = CC (as NeuralHub's default), 1 = HT. NeuralHub picks
        // its model in neuralhub.ini.
        int model = 0;
        // NVIDIA: feature 18 runs this many times in a row (1..kMaxNrPasses), each pass on the
        // one before's output with its own feature, temporal history and NrTuning.
        int passes = 1;
        bool operator==(const NrConfig &o) const
        {
            return width == o.width && height == o.height && out_width == o.out_width && out_height == o.out_height
                && hdr == o.hdr && model == o.model && passes == o.passes;
        }
    };
    // Whether this GPU can upscale after neural rendering (NeuralHub: always; NVIDIA: with
    // nvngx_dlss.dll next to chiaki.exe).
    bool nrCanUpscale() const;
    // nrEnsure warms the model up (a few seconds). No GPU work may be pending on its buffers.
    bool nrEnsure(const NrConfig &config);
    bool nrMatches(const NrConfig &config) const;
    // Whether the last failed nrEnsure() failed in the upscaling part of an upscaling config (DLSS
    // Super Resolution's creation or warm-up evaluation, or the output-size buffers), not in neural
    // rendering itself: the same config without upscaling (out = in) may well work.
    bool nrUpscaleFailed() const;
    // As invalidate(), for the neural rendering state.
    void nrInvalidate();
    // Copies between Vulkan images (VK_IMAGE_LAYOUT_GENERAL, VK_FORMAT_R8G8B8A8_UNORM or
    // VK_FORMAT_R16G16B16A16_SFLOAT for HDR) and the shared input buffer (width x height) /
    // output buffer (out_width x out_height).
    void recordNrInput(VkCommandBuffer cmd, VkImage image);
    void recordNrOutput(VkCommandBuffer cmd, VkImage image);
    // Waits (on the GPU) for `wait_value`, runs the feature on the input buffer, writes the
    // output buffer and signals `signal_value` (always, as generate() does).
    bool neuralRender(uint64_t wait_value, uint64_t signal_value, bool reset);
    // Average GPU time of the neural rendering evaluation in milliseconds (0 if unknown),
    // upscaling and the HDR proxy passes included.
    float nrGpuTimeMs() const;

    // Live controls of NVIDIA's feature 18 (its DLSSNR.* evaluation keys). They are set with
    // every evaluation, so a change needs no new feature; the evaluation after a change starts
    // a new temporal history. NeuralHub sets these in its own panel (nrProviderPanel).
    struct NrTuning
    {
        int style = 0;                // 0 Default, 1 Natural, 2 Cinematic
        float intensity = 1.0f;       // 0..1
        float local_structure = 1.0f; // 0..2
        float local_tone = 1.0f;      // 0..2
        bool auto_mask = false;       // the model finds skin (and masks it) itself
        float skin_structure = 1.0f;  // 0..2.5, only with auto_mask (1 = the model's own: the
                                      // snippet gives negative values the same result)
        bool operator==(const NrTuning &o) const
        {
            return style == o.style && intensity == o.intensity && local_structure == o.local_structure
                && local_tone == o.local_tone && auto_mask == o.auto_mask && skin_structure == o.skin_structure;
        }
        bool operator!=(const NrTuning &o) const { return !(*this == o); }
    };
    static constexpr int kMaxNrPasses = 3;
    // The controls of one pass (0..kMaxNrPasses-1), from the next neuralRender() on (call it
    // on the same thread). Values are clamped. A change also starts a new history in the
    // passes after it (their input changed).
    void nrSetTuning(int pass, const NrTuning &tuning);
    // Whether NrTuning reaches the model (NVIDIA's feature 18 route).
    bool nrTunable() const;

    // NeuralHub's in-game panel (its key, End by default), as in games. NeuralHub polls its
    // keys and draws the panel when its swapchain is presented, so on the NeuralHub route the
    // hidden swapchain is presented after every neuralRender(). NeuralHub logs when the panel
    // opens and closes (dlssnr_on_amd.log next to the executable); while it is open and
    // `window` (chiaki's top-level window, an HWND) has the focus, the swapchain's window is
    // shown over the video in that window, every frame copied into it under the panel.
    // NeuralHub keeps the panel's keys from the process's message loop while it is open.
    void nrSetPanelWindow(void *window);
    // Whether neural rendering runs through NeuralHub, which then owns its panel key.
    bool nrProviderPanel() const;
    // Hides the panel's window (neural rendering paused or stopped; it comes back with the
    // next neuralRender() while NeuralHub's panel is still open).
    void nrHidePanel();

    // One instance per Vulkan device, shared by frame generation and neural rendering
    // (one D3D12 device and one NGX runtime for both). Created and initialised on first
    // use; nullptr when init() fails. The log callback of the first caller is used.
    static std::shared_ptr<Dx12Fg> shared(const VulkanDevice &vk, LogFn log, void *user);
    // True while an instance in this process has NGX's D3D12 side initialised. Once it is up,
    // NGX refuses to create DLSS Frame Generation through Vulkan (InvalidParameter), so DLSS FG
    // has to run through this bridge too.
    static bool ngxUpInProcess();
    // After a crash inside the runtime its state is unknown: the user that saw it calls
    // markCrashed() and leaks its reference (the instance is never destroyed), and
    // shared() returns nullptr for that Vulkan device from then on. A fault inside
    // nvngx_dlssnr.dll, which nrgate turns into an error code, marks it crashed by itself: check
    // crashed() after a failed neuralRender() / nrEnsure() / nrInvalidate().
    void markCrashed();
    bool crashed() const;
    // D3D12 work on the shared queue did not finish (a host wait gave up, or the renderer's bounded
    // wait for Vulkan work behind it timed out): the user that saw it calls markStuck(), the other
    // user stops at its next call, and shared() returns nullptr from then on. Host waits for the
    // D3D12 queue give up after 250 ms instead of 15 s from then on (until one succeeds).
    void markStuck();
    bool stuck() const;
    // Raises the shared timeline semaphore to `value` from the host when its counter is below it.
    // After a crash inside the runtime the D3D12 call that would have signalled `value` never ran,
    // and Vulkan work waiting for it on the GPU (holding up everything queued after it on the same
    // queue) would wait forever. It first waits up to 1 s for `value - 1` (the Vulkan copy in front
    // of the D3D12 call, which itself waited for every earlier D3D12 signal), so no earlier signal
    // lands after this one and moves the counter back. Vulkan calls only (never D3D12 or the
    // runtime): safe after markCrashed(). Call it with lastValue().
    void signalFromHost(uint64_t value);
    // For D3D12 work that is stuck on the GPU (it may never signal): raises the shared timeline
    // semaphore to `value` (call it with lastValue()) from the host and keeps it there until the
    // caller's own Vulkan work is through (`done` reaches `done_value`, e.g. a user's own timeline of
    // its submissions) or `timeout_ms` passed. Vulkan copies queued behind a wait that only the stuck
    // D3D12 work would have satisfied still signal their lower values on the shared timeline as they
    // run, which moves its counter back below the waits after them: it is raised again each time.
    // Vulkan calls only. True when `done` reached `done_value`.
    bool raiseUntilDone(uint64_t value, VkSemaphore done, uint64_t done_value, int timeout_ms);
    // How many host waits for D3D12 GPU work gave up (15 s) since the instance was created.
    uint32_t gpuWaitTimeouts() const;

    // The runtime call in progress, named in the log if the runtime crashes.
    const char *stage() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_DX12FG_H
