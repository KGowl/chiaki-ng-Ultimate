#ifndef CHIAKI_GUI_NEURALRENDER_H
#define CHIAKI_GUI_NEURALRENDER_H

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "neuralrendertuning.h"

#include <array>
#include <cstdint>
#include <memory>

// DLSS 5 Neural Rendering on the video, through the NGX D3D12 bridge (dx12fg.h).
//
// Each new stream frame is rendered into a texture at its own size (RGBA8 sRGB, or RGBA16F
// linear light for HDR streams: 1.0 = SDR white, BT.709 primaries), copied into a buffer
// shared with D3D12, run through DLSS 5 (and upscaled when asked) and copied back into an
// RGB texture that replaces the frame.
//
//  - NVIDIA: NGX feature 18 from nvngx_dlssnr.dll next to chiaki.exe (not bundled) at the
//    stream size, then NVIDIA's DLSS Super Resolution (nvngx_dlss.dll) to upscale. The model
//    is SDR only, so an HDR frame goes through an SDR proxy whose change is put back on it.
//  - Other GPUs: one DLSS Super Resolution feature (DLAA at 1:1, a quality mode when
//    upscaling) that NeuralHub (its dxgi.dll stub and neuralhub.ini next to chiaki.exe)
//    serves with its DLSS 5 neural rendering pass in front of the upscaler, the route the
//    DLSS 5 video players take. HDR is passed as HDR.
//
// Only functional in builds with CHIAKI_HAVE_DX12FG (Windows, NGX SDK); otherwise every
// call is a no-op that reports it as unavailable.
class NeuralRender
{
public:
    static bool compiledIn();

    explicit NeuralRender(pl_log log);
    ~NeuralRender();
    NeuralRender(const NeuralRender &) = delete;
    NeuralRender &operator=(const NeuralRender &) = delete;

    // Lazily creates the D3D12 bridge (shared with DLSS Frame Generation through D3D12).
    // After a failure this keeps returning false without retrying.
    bool ensureInitialized(pl_vulkan vk);
    bool failed() const;
    // Whether the frames can be upscaled after neural rendering (NVIDIA: needs
    // nvngx_dlss.dll next to chiaki.exe).
    bool canUpscale() const;
    // Average GPU time of the neural rendering evaluation (D3D12 side, upscaling included)
    // in milliseconds, 0 if unknown.
    float gpuTimeMs() const;

    // Waits for in-flight work and releases every resource. Must be called before the
    // pl_vulkan device is destroyed.
    void destroy();

    // The next processed frame starts a new temporal history (after neural rendering was
    // paused, so the model does not blend in frames from before the pause).
    void resetHistory();

    // NeuralHub's in-game panel (End), shown over the video like in games (dx12fg.h):
    // `window` is chiaki's top-level window (an HWND on Windows). hasProviderPanel() says
    // whether NeuralHub runs the neural rendering (it then owns the End key); any thread.
    // hidePanel() while neural rendering is paused.
    void setPanelWindow(void *window);
    bool hasProviderPanel() const;
    void hidePanel();
    // Whether Options::passes and Options::tuning reach the model (NVIDIA's feature 18);
    // any thread.
    bool tunable() const;

    struct Options
    {
        // Output size; 0 (or not larger than the frame) keeps the frame's own size.
        int out_width = 0;
        int out_height = 0;
        // NVIDIA feature 18 model: 0 = CC, 1 = HT. NeuralHub uses its own setting.
        int model = 0;
        // NVIDIA: feature 18 passes in a row (1..3; changing it re-creates the features) and
        // the live controls of each pass (applied from this frame on). NeuralHub has its own.
        int passes = 1;
        std::array<NeuralRenderTuning, NeuralRenderTuning::kMaxPasses> tuning = {};
        // How many of the frames process() returns the caller uses at once: 1 with Direct Mapping
        // (one frame per render), the number of frames mixed with the frame mixer. The cache then
        // keeps that many plus one rendered frames (at least 2) instead of up to 8 (at a 4K HDR
        // output 66 MB each). 0 = not known: up to 8, or 2 while frames come through begin().
        int frames_in_use = 0;
    };

    // Runs neural rendering on `src`. On success `out_frame` describes an RGB texture owned
    // by this object, valid at least until `process` is called for more distinct frames than
    // the internal cache holds (Options::frames_in_use + 1, at least 2; 8 when not given); its
    // crop is the output size. The first frame of a size or setting waits for the model to warm
    // up (a few seconds). When DLSS Super Resolution cannot be set up for an upscaling size, that
    // size is not asked for again: the frame is rendered at its own size (one log line).
    bool process(const struct pl_frame *src, uint64_t signature, const Options &options,
                 struct pl_frame *out_frame, uint64_t *out_signature);

    // Pipelined use (frame generation): starts neural rendering of `src` like process() but
    // returns once the D3D12 work is queued, without queueing the copy back into the Vulkan
    // texture (Vulkan work that waits for D3D12 holds up everything queued after it, like
    // the frames presented meanwhile). beginDone() says whether the D3D12 side finished.
    // process() with the same frame, signature and options then returns the result; it
    // queues the copy back, which waits on the GPU if the D3D12 side is not done yet.
    // Starting another frame drops a begun frame nobody collected.
    bool begin(const struct pl_frame *src, uint64_t signature, const Options &options);
    bool beginDone() const;

    // Render thread, after a bounded wait for Vulkan work timed out (the swapchain image, or a
    // fence of the work queued for a render). Neural rendering's Vulkan copies wait on the GPU for
    // its D3D12 work; when that work is behind on the GPU (stuck), this raises the shared D3D12
    // timeline to the last value handed out from the host and keeps it there until neural
    // rendering's own Vulkan work is through (at most about 1 s), then turns neural rendering off
    // for the session: failed() is true from then on. What the GPU may still use (textures,
    // command buffers, semaphores, the D3D12 buffers) is leaked, not freed; frames handed out
    // earlier stay valid. Does nothing when the D3D12 side is not behind (the stall is elsewhere)
    // or neural rendering is not running. Call FrameGen::unstickGpu() too, then retry the wait once.
    void unstickGpu();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_NEURALRENDER_H
