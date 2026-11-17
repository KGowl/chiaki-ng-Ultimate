#include "ffxfi.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>

#include <QLoggingCategory>

#include <array>
#include <vector>

Q_DECLARE_LOGGING_CATEGORY(chiakiGui);

namespace {

// Optical flow, frame interpolation and the shared resources; the backend
// sizes its descriptor pools by this.
constexpr size_t kMaxContexts = 3;
// Video has no depth: a constant mid-range value, as for the other backends.
constexpr float kCameraNear = 0.1f;
constexpr float kCameraFar = 1000.0f;

FfxResource wrapImage(const FfxFiRunner::Image &img, const wchar_t *name, FfxResourceUsage usage)
{
    FfxResourceDescription desc = {};
    desc.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    desc.format = ffxGetSurfaceFormatVK(img.format);
    desc.width = static_cast<uint32_t>(img.w);
    desc.height = static_cast<uint32_t>(img.h);
    desc.depth = 1;
    desc.mipCount = 1;
    desc.flags = FFX_RESOURCE_FLAGS_NONE;
    desc.usage = usage;
    // GENERAL layout is the backend's UAV state; it restores what it was given.
    return ffxGetResourceVK(reinterpret_cast<void *>(img.image), desc, name, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
}

} // namespace

struct FfxFiRunner::Impl
{
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    const char *stage = "";

    std::vector<uint8_t> scratch;
    FfxInterface iface = {};
    bool iface_ok = false;
    FfxOpticalflowContext of = {};
    FfxFrameInterpolationContext fi = {};
    bool of_ok = false;
    bool fi_ok = false;
    uint32_t shared_id = 0;
    bool shared_ok = false;
    std::array<FfxResourceInternal, 5> shared = {}; // OF vector, OF SCD, dilated depth, dilated MV, prev depth
    bool shared_res_ok = false;
    bool shared_transitioned = false;
    uint32_t of_w = 0;
    uint32_t of_h = 0;
    int width = 0;
    int height = 0;
    uint64_t frame_id = 0;

    bool check(FfxErrorCode code, const char *what)
    {
        if (code == FFX_OK)
            return true;
        qCWarning(chiakiGui) << "[framegen] FSR (FidelityFX SDK)" << what << "failed, error" << static_cast<int>(code);
        return false;
    }
    void destroyContexts();
};

void FfxFiRunner::Impl::destroyContexts()
{
    if (fi_ok)
        ffxFrameInterpolationContextDestroy(&fi);
    if (of_ok)
        ffxOpticalflowContextDestroy(&of);
    fi_ok = of_ok = false;
    if (shared_res_ok && shared_ok) {
        for (FfxResourceInternal r : shared)
            iface.fpDestroyResource(&iface, r, shared_id);
    }
    shared_res_ok = false;
    shared_transitioned = false;
    if (shared_ok)
        iface.fpDestroyBackendContext(&iface, shared_id);
    shared_ok = false;
    width = height = 0;
}

FfxFiRunner::FfxFiRunner(VkDevice device, VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr get_device_proc)
    : impl(std::make_unique<Impl>())
{
    impl->device = device;
    impl->physical = physical;
    impl->get_device_proc = get_device_proc;
    impl->vkCmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get_device_proc(device, "vkCmdPipelineBarrier"));
}

FfxFiRunner::~FfxFiRunner()
{
    impl->destroyContexts();
}

const char *FfxFiRunner::stage() const
{
    return impl->stage;
}

void FfxFiRunner::invalidate()
{
    impl->destroyContexts();
}

bool FfxFiRunner::ensure(int w, int h)
{
    Impl &d = *impl;
    if (d.fi_ok && w == d.width && h == d.height)
        return true;
    d.destroyContexts();
    if (!d.vkCmdPipelineBarrier)
        return false;

    VkDeviceContext device_context = { d.device, d.physical, d.get_device_proc };
    if (!d.iface_ok) {
        d.stage = "ffxGetInterfaceVK";
        d.scratch.assign(ffxGetScratchMemorySizeVK(d.physical, kMaxContexts), 0);
        if (!d.check(ffxGetInterfaceVK(&d.iface, ffxGetDeviceVK(&device_context), d.scratch.data(), d.scratch.size(),
                                       kMaxContexts), "ffxGetInterfaceVK"))
            return false;
        d.iface_ok = true;
    }
    d.iface.device = ffxGetDeviceVK(&device_context);

    d.stage = "ffxOpticalflowContextCreate";
    FfxOpticalflowContextDescription of_desc = {};
    of_desc.backendInterface = d.iface;
    of_desc.resolution = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    if (!d.check(ffxOpticalflowContextCreate(&d.of, &of_desc), "ffxOpticalflowContextCreate"))
        return false;
    d.of_ok = true;

    d.stage = "ffxFrameInterpolationContextCreate";
    FfxFrameInterpolationContextDescription fi_desc = {};
    fi_desc.backendInterface = d.iface;
    fi_desc.flags = FFX_FRAMEINTERPOLATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    fi_desc.maxRenderSize = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
    fi_desc.displaySize = fi_desc.maxRenderSize;
    fi_desc.backBufferFormat = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    fi_desc.previousInterpolationSourceFormat = FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    if (!d.check(ffxFrameInterpolationContextCreate(&d.fi, &fi_desc), "ffxFrameInterpolationContextCreate")) {
        d.destroyContexts();
        return false;
    }
    d.fi_ok = true;

    d.stage = "shared resources";
    if (!d.check(d.iface.fpCreateBackendContext(&d.iface, FFX_EFFECT_SHAREDRESOURCES, nullptr, &d.shared_id),
                 "shared resource context")) {
        d.destroyContexts();
        return false;
    }
    d.shared_ok = true;
    FfxOpticalflowSharedResourceDescriptions of_res = {};
    FfxFrameInterpolationSharedResourceDescriptions fi_res = {};
    ffxOpticalflowGetSharedResourceDescriptions(&d.of, &of_res);
    ffxFrameInterpolationGetSharedResourceDescriptions(&d.fi, &fi_res);
    const FfxCreateResourceDescription *descs[] = { &of_res.opticalFlowVector, &of_res.opticalFlowSCD,
                                                    &fi_res.dilatedDepth, &fi_res.dilatedMotionVectors,
                                                    &fi_res.reconstructedPrevNearestDepth };
    for (size_t i = 0; i < d.shared.size(); i++) {
        if (!d.check(d.iface.fpCreateResource(&d.iface, descs[i], d.shared_id, &d.shared[i]), "shared resource creation")) {
            // The ones created so far are released with the backend context.
            d.destroyContexts();
            return false;
        }
    }
    d.shared_res_ok = true;
    d.of_w = of_res.opticalFlowVector.resourceDescription.width;
    d.of_h = of_res.opticalFlowVector.resourceDescription.height;
    d.width = w;
    d.height = h;
    d.frame_id = 0;
    return true;
}

bool FfxFiRunner::record(VkCommandBuffer cmd, const Image &color, const Image &output, const Image &motion,
                         const Image &depth, bool reset, float frame_time_ms)
{
    Impl &d = *impl;
    if (!d.fi_ok || color.w != d.width || color.h != d.height)
        return false;
    const FfxCommandList cl = ffxGetCommandListVK(cmd);

    std::array<FfxResource, 5> shared;
    for (size_t i = 0; i < shared.size(); i++)
        shared[i] = d.iface.fpGetResource(&d.iface, d.shared[i]);
    if (!d.shared_transitioned) {
        // The backend treats freshly created resources as usable; move them
        // out of UNDEFINED before the first pass reads them.
        std::vector<VkImageMemoryBarrier> barriers;
        for (const FfxResource &res : shared) {
            if (res.description.type == FFX_RESOURCE_TYPE_BUFFER || !res.resource)
                continue;
            VkImageMemoryBarrier b = {};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = reinterpret_cast<VkImage>(res.resource);
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
            b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
            barriers.push_back(b);
        }
        if (!barriers.empty())
            d.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                   0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
        d.shared_transitioned = true;
        reset = true;
    }
    if (d.frame_id == 0)
        reset = true;
    d.frame_id++;
    const float frame_ms = frame_time_ms > 0.0f ? frame_time_ms : 16.6667f;

    const FfxResource cur = wrapImage(color, L"chiaki_fg_color", FFX_RESOURCE_USAGE_UAV);

    d.stage = "ffxOpticalflowContextDispatch";
    FfxOpticalflowDispatchDescription of = {};
    of.commandList = cl;
    of.color = cur;
    of.opticalFlowVector = shared[0];
    of.opticalFlowSCD = shared[1];
    of.reset = reset;
    of.backbufferTransferFunction = FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    of.minMaxLuminance = { 0.0f, 1.0f };
    if (!d.check(ffxOpticalflowContextDispatch(&d.of, &of), "ffxOpticalflowContextDispatch"))
        return false;

    d.stage = "ffxFrameInterpolationPrepare";
    FfxFrameInterpolationPrepareDescription prep = {};
    prep.commandList = cl;
    prep.renderSize = { static_cast<uint32_t>(color.w), static_cast<uint32_t>(color.h) };
    prep.jitterOffset = { 0.0f, 0.0f };
    prep.motionVectorScale = { 1.0f, 1.0f }; // pixels
    prep.frameTimeDelta = frame_ms;
    prep.cameraNear = kCameraNear;
    prep.cameraFar = kCameraFar;
    prep.viewSpaceToMetersFactor = 1.0f;
    prep.cameraFovAngleVertical = 1.0f;
    prep.depth = wrapImage(depth, L"chiaki_fg_depth", FFX_RESOURCE_USAGE_READ_ONLY);
    prep.motionVectors = wrapImage(motion, L"chiaki_fg_motion", FFX_RESOURCE_USAGE_READ_ONLY);
    prep.frameID = d.frame_id;
    prep.dilatedDepth = shared[2];
    prep.dilatedMotionVectors = shared[3];
    prep.reconstructedPrevDepth = shared[4];
    if (!d.check(ffxFrameInterpolationPrepare(&d.fi, &prep), "ffxFrameInterpolationPrepare"))
        return false;

    d.stage = "ffxFrameInterpolationDispatch";
    FfxFrameInterpolationDispatchDescription fi = {};
    fi.commandList = cl;
    fi.displaySize = prep.renderSize;
    fi.renderSize = prep.renderSize;
    fi.currentBackBuffer = cur;
    fi.output = wrapImage(output, L"chiaki_fg_output", FFX_RESOURCE_USAGE_UAV);
    fi.interpolationRect = { 0, 0, color.w, color.h };
    fi.opticalFlowVector = shared[0];
    fi.opticalFlowSceneChangeDetection = shared[1];
    fi.opticalFlowBufferSize = { d.of_w, d.of_h };
    fi.opticalFlowScale = { 1.0f / static_cast<float>(color.w), 1.0f / static_cast<float>(color.h) };
    fi.opticalFlowBlockSize = 8;
    fi.cameraNear = kCameraNear;
    fi.cameraFar = kCameraFar;
    fi.cameraFovAngleVertical = 1.0f;
    fi.viewSpaceToMetersFactor = 1.0f;
    fi.frameTimeDelta = frame_ms;
    fi.reset = reset;
    fi.backBufferTransferFunction = FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    fi.minMaxLuminance[0] = 0.0f;
    fi.minMaxLuminance[1] = 1.0f;
    fi.frameID = d.frame_id;
    fi.dilatedDepth = shared[2];
    fi.dilatedMotionVectors = shared[3];
    fi.reconstructedPrevDepth = shared[4];
    if (!d.check(ffxFrameInterpolationDispatch(&d.fi, &fi), "ffxFrameInterpolationDispatch"))
        return false;
    d.stage = "";
    return true;
}
