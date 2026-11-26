#ifndef CHIAKI_GUI_FRAMEGEN_H
#define CHIAKI_GUI_FRAMEGEN_H

#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include <memory>
#include <vector>

// Frame generation (interpolation) for the video stream on libplacebo's
// Vulkan device. Each real frame is rendered to an RGBA8 texture at display
// size (HDR streams: RGBA16F holding BT.2020 PQ values, see backendHdrCapable)
// and handed to a backend, which interpolates between it and the previous
// real frame. Video has no motion vectors or depth, so the backends
// get zero motion vectors and a constant depth and rely on their own optical
// flow.
//
// Backends (each only functional when compiled in):
//  - DLSS Frame Generation through NGX (CHIAKI_HAVE_NGX), RTX 40 series or
//    newer, up to 3 generated frames per real frame on RTX 50 series.
//  - AMD FidelityFX FSR 3.1 frame generation (CHIAKI_HAVE_FFX), any GPU,
//    1 generated frame per real frame.
//  - Flow: chiaki-ng's own optical flow interpolation (FlowInterpolator),
//    built from libplacebo compute shaders. Always available, any GPU and
//    OS (Windows, Linux, Steam Deck), up to 3 generated frames per real frame.
class FrameGen
{
public:
    enum class Backend {
        None = 0,
        Dlss = 1,
        Fsr = 2,
        Flow = 3,
        // NVIDIA Video Frame Generation (Video Effects SDK), Windows only.
        Vfg = 4,
        // DLSS Frame Generation through the NGX D3D12 API (dx12fg.h), Windows only. Meant
        // for GPUs without NVIDIA's runtime that get an NGX provider next to chiaki.exe
        // (NeuralHub's dxgi.dll stub); Vulkan frames are shared with a D3D12 device.
        Dlss12 = 5,
    };

    static bool compiledIn();
    static bool backendCompiledIn(Backend backend);
    // Whether `backend` generates frames for HDR streams (DLSS FG through Vulkan or D3D12, Flow).
    // FSR 3.1 and NVIDIA VFG are 8-bit only: process() returns false for HDR frames with them.
    static bool backendHdrCapable(Backend backend);
    // NGX's D3D12 side is up in this process (DLSS 5 or DLSS FG through the D3D12 bridge). NGX
    // then refuses DLSS Frame Generation through Vulkan, so Backend::Dlss has to run as Dlss12.
    static bool d3d12NgxUp();
    static const char *backendName(Backend backend);

    explicit FrameGen(pl_log log);
    ~FrameGen();
    FrameGen(const FrameGen &) = delete;
    FrameGen &operator=(const FrameGen &) = delete;

    // Lazily initializes `backend`; switching to a different backend tears the
    // previous one down first. After a failure the same backend is not retried.
    bool ensureInitialized(pl_vulkan vk, Backend backend);
    Backend backend() const;
    // Whether `backend` failed to start or crashed and is not retried.
    bool backendFailed(Backend backend) const;

    // Highest output multiplier the active backend supports (2 = one
    // generated frame per real frame).
    int maxMultiplier() const;

    // Average GPU time of the backend's generation work per stream frame in
    // milliseconds (0 if unknown).
    float gpuTimeMs() const;

    // The next frame starts a new sequence: nothing is interpolated across it.
    void reset();

    // Flow backend only: analyse motion at 1/2 instead of 1/4 resolution.
    void setFlowHighQuality(bool high);
    // DLSS Frame Generation motion vectors: 0 zero motion, 1 Flow's motion
    // search (default), 2 the same with the opposite sign. Uses the Flow quality.
    void setDlssMotion(int mode);
    // NVIDIA VFG model: 0 Low, 1 Medium, 2 High.
    void setVfgMode(int mode);
    // Clean-up of the generated frames (see fgfixup.h): 0 off, 1 static
    // content (HUD, text) only, 2 static content and fine detail (default).
    void setCleanup(int mode);

    // Waits for in-flight work and releases every Vulkan and backend resource.
    // Must be called before the pl_vulkan device is destroyed.
    void destroy();

    // Renders `src` at `out_w` x `out_h` and generates `multiplier - 1` frames
    // between the previous real frame and this one. `outputs` receives the
    // frames in presentation order: generated frames first, the real frame
    // last; without a previous frame only the real frame is returned. The
    // textures stay valid until `process` has been called twice more.
    bool process(const struct pl_frame *src, int out_w, int out_h, int multiplier,
                 float frame_interval_ms, std::vector<struct pl_frame> &outputs);

    // Pipelined use, DLSS Frame Generation through D3D12 only (other backends return false,
    // use process()): begin() does what process() does up to queueing the D3D12 generation
    // and returns without queueing the copy back of the generated frames, so the frames
    // presented meanwhile are not held up behind Vulkan work waiting for D3D12.
    // beginDone() says whether the D3D12 side finished; finish() then fills `outputs` as
    // process() would (it queues the copy back, which waits on the GPU if needed). A begun
    // call that is never finished is dropped by the next begin(), process() or reset().
    bool begin(const struct pl_frame *src, int out_w, int out_h, int multiplier, float frame_interval_ms);
    bool beginDone() const;
    bool finish(std::vector<struct pl_frame> &outputs);

    struct Impl;

private:
    bool processFrames(const struct pl_frame *src, int out_w, int out_h, int multiplier,
                       float frame_interval_ms, std::vector<struct pl_frame> &outputs, bool defer = false);

    std::unique_ptr<Impl> impl;
};

#endif // CHIAKI_GUI_FRAMEGEN_H
