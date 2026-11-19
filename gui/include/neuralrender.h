#ifndef CHIAKI_GUI_NEURALRENDER_H
#define CHIAKI_GUI_NEURALRENDER_H

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

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

    struct Options
    {
        // Output size; 0 (or not larger than the frame) keeps the frame's own size.
        int out_width = 0;
        int out_height = 0;
        // NVIDIA feature 18 model: 0 = CC, 1 = HT. NeuralHub uses its own setting.
        int model = 0;
    };

    // Runs neural rendering on `src`. On success `out_frame` describes an RGB texture owned
    // by this object, valid at least until `process` is called for more distinct frames than
    // the internal cache holds; its crop is the output size. The first frame of a size or
    // setting waits for the model to warm up (a few seconds).
    bool process(const struct pl_frame *src, uint64_t signature, const Options &options,
                 struct pl_frame *out_frame, uint64_t *out_signature);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_NEURALRENDER_H
