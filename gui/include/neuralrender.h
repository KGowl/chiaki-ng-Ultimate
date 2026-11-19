#ifndef CHIAKI_GUI_NEURALRENDER_H
#define CHIAKI_GUI_NEURALRENDER_H

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include <cstdint>
#include <memory>

// DLSS 5 Neural Rendering on the video, through the NGX D3D12 bridge (dx12fg.h).
//
// Each new stream frame is rendered into an RGBA8 texture at its own size, copied into a
// buffer shared with D3D12, run through a DLSS Super Resolution feature in DLAA mode
// (input size = output size) and copied back into an RGB texture that replaces the frame.
// On a GPU without NVIDIA's runtime, NeuralHub (its dxgi.dll stub and neuralhub.ini next to
// chiaki.exe) serves that call and runs its DLSS 5 neural rendering pass first; this is the
// route the DLSS 5 video players take. NVIDIA's own runtime gives plain DLAA instead.
//
// Only functional in builds with CHIAKI_HAVE_DX12FG (Windows, NGX SDK); otherwise every
// call is a no-op that reports it as unavailable. SDR only (8-bit), like RTX VSR.
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
    // Average GPU time of the neural rendering evaluation (D3D12 side) in milliseconds,
    // 0 if unknown.
    float gpuTimeMs() const;

    // Waits for in-flight work and releases every resource. Must be called before the
    // pl_vulkan device is destroyed.
    void destroy();

    // Runs neural rendering on `src` at its own size. On success `out_frame` describes an
    // RGB texture owned by this object, valid at least until `process` is called for more
    // distinct frames than the internal cache holds. The first frame of a size waits for
    // the model to warm up (a few seconds).
    bool process(const struct pl_frame *src, uint64_t signature, struct pl_frame *out_frame,
                 uint64_t *out_signature);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_NEURALRENDER_H
