#include "neuralrender.h"

#include <QLoggingCategory>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

Q_DECLARE_LOGGING_CATEGORY(chiakiGui);

#if defined(CHIAKI_HAVE_DX12FG)

#include "dx12fg.h"
#include "vkguard.h"

namespace {

// Rendered frames are cached per source frame so the frame mixer can reuse them.
constexpr size_t kMaxCachedFrames = 8;
// Two command buffers per frame (copy in, copy out).
constexpr size_t kCommandSlots = 6;
constexpr uint64_t kWaitTimeoutNs = 2000000000ull;
constexpr int kMaxFailures = 3;
constexpr VkImageLayout kLayout = VK_IMAGE_LAYOUT_GENERAL;

void bridgeLog(int level, const char *message, void *)
{
    if (level > 0)
        qCWarning(chiakiGui) << "[neural-render] D3D12:" << message;
    else
        qCInfo(chiakiGui) << "[neural-render] D3D12:" << message;
}

} // namespace

struct NeuralRender::Impl
{
    struct Image
    {
        pl_tex tex = nullptr;
        VkImage image = VK_NULL_HANDLE;
        int w = 0;
        int h = 0;
    };

    struct CacheEntry
    {
        Image image;
        uint64_t key = 0;
        uint64_t last_use = 0;
        uint64_t done_value = 0;
    };

    struct CommandSlot
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        uint64_t done_value = 0;
    };

    pl_log log = nullptr;
    pl_vulkan vk = nullptr;
    pl_gpu gpu = nullptr;
    pl_renderer renderer = nullptr;
    pl_fmt fmt = nullptr;
    bool initialized = false;
    bool init_failed = false;
    bool hdr_warned = false;
    bool reset_pending = true;
    int failures = 0;
    std::shared_ptr<Dx12Fg> runner;

    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::array<CommandSlot, kCommandSlots> cmd_slots = {};
    size_t next_slot = 0;

    VkSemaphore sem_in = VK_NULL_HANDLE;
    VkSemaphore sem_out = VK_NULL_HANDLE;
    VkSemaphore sem_done = VK_NULL_HANDLE;
    uint64_t in_value = 0;
    uint64_t out_value = 0;
    uint64_t done_value = 0;

    Image input;
    std::vector<CacheEntry> cache;
    uint64_t use_counter = 0;

    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkWaitSemaphores vkWaitSemaphores = nullptr;

    bool loadFunctions();
    bool init(pl_vulkan vk);
    void shutdown();
    bool waitDone(uint64_t value);
    CommandSlot *beginSlot();
    bool submit(VkCommandBuffer cmd, VkSemaphore image_sem, uint64_t image_value, uint64_t shared_wait,
                uint64_t shared_signal);
    bool ensureImage(Image &img, int w, int h);
    void destroyImage(Image &img);
    CacheEntry &acquireCacheEntry(uint64_t key);
    bool run(Image &out, int w, int h, bool reset, bool *crashed);
    void abandonRunner();
};

bool NeuralRender::Impl::loadFunctions()
{
    vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        vk->get_proc_addr(vk->instance, "vkGetDeviceProcAddr"));
    if (!vkGetDeviceProcAddr)
        return false;
#define LOAD_DEVICE_PROC(name_) \
    name_ = reinterpret_cast<PFN_##name_>(vkGetDeviceProcAddr(vk->device, #name_)); \
    if (!name_) { \
        qCWarning(chiakiGui) << "[neural-render] Missing Vulkan function" << #name_; \
        return false; \
    }
    LOAD_DEVICE_PROC(vkGetDeviceQueue)
    LOAD_DEVICE_PROC(vkCreateCommandPool)
    LOAD_DEVICE_PROC(vkDestroyCommandPool)
    LOAD_DEVICE_PROC(vkAllocateCommandBuffers)
    LOAD_DEVICE_PROC(vkBeginCommandBuffer)
    LOAD_DEVICE_PROC(vkEndCommandBuffer)
    LOAD_DEVICE_PROC(vkResetCommandBuffer)
    LOAD_DEVICE_PROC(vkQueueSubmit)
    LOAD_DEVICE_PROC(vkWaitSemaphores)
#undef LOAD_DEVICE_PROC
    return true;
}

bool NeuralRender::Impl::init(pl_vulkan vk_)
{
    vk = vk_;
    gpu = vk->gpu;
    if (!loadFunctions())
        return false;
    fmt = pl_find_named_fmt(gpu, "rgba8");
    const enum pl_fmt_caps needed_caps = static_cast<enum pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE);
    if (!fmt || (fmt->caps & needed_caps) != needed_caps) {
        qCWarning(chiakiGui) << "[neural-render] rgba8 format is not renderable on this device";
        return false;
    }
    queue_family = vk->queue_graphics.index;
    vkGetDeviceQueue(vk->device, queue_family, 0, &queue);
    if (!queue)
        return false;
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    if (vkCreateCommandPool(vk->device, &pool_info, nullptr, &pool) != VK_SUCCESS)
        return false;
    std::array<VkCommandBuffer, kCommandSlots> cmds = {};
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(cmds.size());
    if (vkAllocateCommandBuffers(vk->device, &alloc_info, cmds.data()) != VK_SUCCESS)
        return false;
    for (size_t i = 0; i < cmds.size(); i++)
        cmd_slots[i].cmd = cmds[i];
    struct pl_vulkan_sem_params sem_params = {};
    sem_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
    sem_in = pl_vulkan_sem_create(gpu, &sem_params);
    sem_out = pl_vulkan_sem_create(gpu, &sem_params);
    sem_done = pl_vulkan_sem_create(gpu, &sem_params);
    if (!sem_in || !sem_out || !sem_done)
        return false;
    renderer = pl_renderer_create(log, gpu);
    if (!renderer)
        return false;

    Dx12Fg::VulkanDevice device;
    device.instance = vk->instance;
    device.physical_device = vk->phys_device;
    device.device = vk->device;
    device.get_instance_proc_addr = vk->get_proc_addr;
    unsigned long seh_code = 0;
    guardedCall([&]() {
        runner = Dx12Fg::shared(device, &bridgeLog, nullptr);
        return runner != nullptr;
    }, &seh_code);
    if (seh_code) {
        qCCritical(chiakiGui).nospace() << "[neural-render] The NGX D3D12 runtime crashed while starting (exception 0x"
                                        << Qt::hex << seh_code << ")";
        (void)new std::shared_ptr<Dx12Fg>(std::move(runner));
        return false;
    }
    if (!runner)
        return false;
    if (!runner->nrAvailable()) {
        qCWarning(chiakiGui) << "[neural-render] The NGX D3D12 runtime offers no DLSS Super Resolution on this GPU";
        runner.reset();
        return false;
    }
    return true;
}

bool NeuralRender::Impl::waitDone(uint64_t value)
{
    if (!sem_done || value == 0)
        return true;
    VkSemaphoreWaitInfo wait_info = {};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &sem_done;
    wait_info.pValues = &value;
    const VkResult res = vkWaitSemaphores(vk->device, &wait_info, kWaitTimeoutNs);
    if (res != VK_SUCCESS) {
        qCWarning(chiakiGui) << "[neural-render] Waiting for the copies failed, result" << static_cast<int>(res);
        return false;
    }
    return true;
}

NeuralRender::Impl::CommandSlot *NeuralRender::Impl::beginSlot()
{
    CommandSlot &slot = cmd_slots[next_slot];
    next_slot = (next_slot + 1) % cmd_slots.size();
    if (!waitDone(slot.done_value) || vkResetCommandBuffer(slot.cmd, 0) != VK_SUCCESS)
        return nullptr;
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(slot.cmd, &begin_info) == VK_SUCCESS ? &slot : nullptr;
}

// Waits for the held image and for `shared_wait` on the D3D12 timeline; signals sem_done
// and, when `shared_signal` is not 0, that value on the D3D12 timeline.
bool NeuralRender::Impl::submit(VkCommandBuffer cmd, VkSemaphore image_sem, uint64_t image_value, uint64_t shared_wait,
                                uint64_t shared_signal)
{
    const VkSemaphore wait_sems[] = { image_sem, runner->semaphore() };
    const uint64_t wait_values[] = { image_value, shared_wait };
    const VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
    const uint64_t signal_value = done_value + 1;
    const VkSemaphore signal_sems[] = { sem_done, runner->semaphore() };
    const uint64_t signal_values[] = { signal_value, shared_signal };
    const uint32_t num_signals = shared_signal ? 2 : 1;

    VkTimelineSemaphoreSubmitInfo timeline_info = {};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 2;
    timeline_info.pWaitSemaphoreValues = wait_values;
    timeline_info.signalSemaphoreValueCount = num_signals;
    timeline_info.pSignalSemaphoreValues = signal_values;
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_info;
    submit_info.waitSemaphoreCount = 2;
    submit_info.pWaitSemaphores = wait_sems;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = num_signals;
    submit_info.pSignalSemaphores = signal_sems;

    vk->lock_queue(vk, queue_family, 0);
    const VkResult res = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    vk->unlock_queue(vk, queue_family, 0);
    if (res != VK_SUCCESS) {
        qCWarning(chiakiGui) << "[neural-render] vkQueueSubmit failed, result" << static_cast<int>(res);
        return false;
    }
    done_value = signal_value;
    return true;
}

bool NeuralRender::Impl::ensureImage(Image &img, int w, int h)
{
    if (img.tex && img.w == w && img.h == h)
        return true;
    destroyImage(img);
    struct pl_tex_params tex_params = {};
    tex_params.w = w;
    tex_params.h = h;
    tex_params.format = fmt;
    tex_params.sampleable = true;
    tex_params.renderable = true;
    img.tex = pl_tex_create(gpu, &tex_params);
    if (!img.tex)
        return false;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    img.image = pl_vulkan_unwrap(gpu, img.tex, &vk_format, nullptr);
    if (!img.image || vk_format != VK_FORMAT_R8G8B8A8_UNORM) {
        qCWarning(chiakiGui) << "[neural-render] Unexpected texture format" << static_cast<int>(vk_format);
        destroyImage(img);
        return false;
    }
    img.w = w;
    img.h = h;
    return true;
}

void NeuralRender::Impl::destroyImage(Image &img)
{
    // Callers make sure no work referencing the image is still pending.
    if (img.tex)
        pl_tex_destroy(gpu, &img.tex);
    img = Image();
}

NeuralRender::Impl::CacheEntry &NeuralRender::Impl::acquireCacheEntry(uint64_t key)
{
    if (cache.size() < kMaxCachedFrames) {
        cache.emplace_back();
        cache.back().key = key;
        return cache.back();
    }
    auto oldest = std::min_element(cache.begin(), cache.end(), [](const CacheEntry &a, const CacheEntry &b) {
        return a.last_use < b.last_use;
    });
    oldest->key = key;
    return *oldest;
}

// The runtime crashed: its state is unknown, so it is never called (or destroyed) again.
void NeuralRender::Impl::abandonRunner()
{
    runner->markCrashed();
    (void)new std::shared_ptr<Dx12Fg>(std::move(runner));
}

// (1) Vulkan copies `input` into the shared buffer and signals the D3D12 timeline, (2) the
// D3D12 queue waits for it, runs the feature and signals the next value, (3) Vulkan waits for
// that and copies the shared output buffer into `out`.
bool NeuralRender::Impl::run(Image &out, int w, int h, bool reset, bool *crashed)
{
    *crashed = false;
    if (!runner->nrSizeMatches(w, h)) {
        // The shared buffers are rebuilt: nothing may still use them.
        if (!waitDone(done_value))
            return false;
        qCInfo(chiakiGui).nospace() << "[neural-render] Preparing DLSS neural rendering for " << w << "x" << h
                                    << " (the model warms up, a few seconds)";
        unsigned long seh_code = 0;
        const bool sized = guardedCall([&]() { return runner->nrEnsureSize(w, h); }, &seh_code);
        if (seh_code) {
            qCCritical(chiakiGui).nospace() << "[neural-render] The NGX D3D12 runtime crashed (exception 0x" << Qt::hex
                                            << seh_code << Qt::dec << ", in " << runner->stage() << ")";
            *crashed = true;
            return false;
        }
        if (!sized)
            return false;
        reset = true;
    }

    // (1)
    CommandSlot *slot = beginSlot();
    if (!slot)
        return false;
    struct pl_vulkan_hold_params hold = {};
    hold.tex = input.tex;
    hold.layout = kLayout;
    hold.qf = VK_QUEUE_FAMILY_IGNORED;
    hold.semaphore = { sem_in, in_value + 1 };
    if (!pl_vulkan_hold_ex(gpu, &hold)) {
        vkEndCommandBuffer(slot->cmd);
        return false;
    }
    in_value++;
    runner->recordNrInput(slot->cmd, input.image);
    const uint64_t input_value = runner->lastValue() + 1;
    // Waiting for lastValue: D3D12 has finished with the buffer being overwritten.
    bool ok = vkEndCommandBuffer(slot->cmd) == VK_SUCCESS &&
              submit(slot->cmd, sem_in, in_value, runner->lastValue(), input_value);
    struct pl_vulkan_release_params release = {};
    release.tex = input.tex;
    release.layout = kLayout;
    release.qf = VK_QUEUE_FAMILY_IGNORED;
    release.semaphore = ok ? pl_vulkan_sem{ sem_done, done_value } : pl_vulkan_sem{ sem_in, in_value };
    pl_vulkan_release_ex(gpu, &release);
    if (!ok)
        return false;
    slot->done_value = done_value;
    runner->setLastValue(input_value);

    // (2) Always signals `output_value`.
    const uint64_t output_value = input_value + 1;
    unsigned long seh_code = 0;
    const bool rendered = guardedCall([&]() { return runner->neuralRender(input_value, output_value, reset); },
                                      &seh_code);
    runner->setLastValue(output_value);
    if (seh_code) {
        qCCritical(chiakiGui).nospace() << "[neural-render] The NGX D3D12 runtime crashed (exception 0x" << Qt::hex
                                        << seh_code << Qt::dec << ", in " << runner->stage() << ")";
        *crashed = true;
        return false;
    }
    if (!rendered)
        return false;

    // (3)
    slot = beginSlot();
    if (!slot)
        return false;
    hold.tex = out.tex;
    hold.semaphore = { sem_out, out_value + 1 };
    if (!pl_vulkan_hold_ex(gpu, &hold)) {
        vkEndCommandBuffer(slot->cmd);
        return false;
    }
    out_value++;
    runner->recordNrOutput(slot->cmd, out.image);
    ok = vkEndCommandBuffer(slot->cmd) == VK_SUCCESS && submit(slot->cmd, sem_out, out_value, output_value, 0);
    release.tex = out.tex;
    release.semaphore = ok ? pl_vulkan_sem{ sem_done, done_value } : pl_vulkan_sem{ sem_out, out_value };
    pl_vulkan_release_ex(gpu, &release);
    if (ok)
        slot->done_value = done_value;
    return ok;
}

void NeuralRender::Impl::shutdown()
{
    if (vk && vkWaitSemaphores)
        waitDone(done_value);
    if (runner) {
        // Frame generation may keep the shared instance alive: drop this side's buffers.
        runner->nrInvalidate();
        runner.reset();
    }
    if (gpu) {
        for (auto &entry : cache)
            destroyImage(entry.image);
        destroyImage(input);
    }
    cache.clear();
    if (pool)
        vkDestroyCommandPool(vk->device, pool, nullptr);
    pool = VK_NULL_HANDLE;
    cmd_slots = {};
    if (gpu) {
        pl_vulkan_sem_destroy(gpu, &sem_in);
        pl_vulkan_sem_destroy(gpu, &sem_out);
        pl_vulkan_sem_destroy(gpu, &sem_done);
    }
    pl_renderer_destroy(&renderer);
    in_value = out_value = done_value = 0;
    initialized = false;
    reset_pending = true;
    vk = nullptr;
    gpu = nullptr;
}

bool NeuralRender::compiledIn()
{
    return true;
}

NeuralRender::NeuralRender(pl_log log)
    : impl(std::make_unique<Impl>())
{
    impl->log = log;
}

NeuralRender::~NeuralRender()
{
    destroy();
}

bool NeuralRender::ensureInitialized(pl_vulkan vk)
{
    if (impl->initialized)
        return true;
    if (impl->init_failed || !vk)
        return false;
    if (!impl->init(vk)) {
        qCWarning(chiakiGui) << "[neural-render] DLSS neural rendering is unavailable (on AMD it needs NeuralHub's "
                                "dxgi.dll and neuralhub.ini next to chiaki.exe)";
        impl->shutdown();
        impl->init_failed = true;
        return false;
    }
    impl->initialized = true;
    qCInfo(chiakiGui) << "[neural-render] DLSS neural rendering bridge ready (NGX D3D12)";
    return true;
}

bool NeuralRender::failed() const
{
    return impl->init_failed;
}

float NeuralRender::gpuTimeMs() const
{
    return impl->runner ? impl->runner->nrGpuTimeMs() : 0.0f;
}

void NeuralRender::destroy()
{
    impl->shutdown();
}

bool NeuralRender::process(const struct pl_frame *src, uint64_t signature, struct pl_frame *out_frame,
                           uint64_t *out_signature)
{
    Impl &d = *impl;
    if (!d.initialized || !src || src->num_planes <= 0)
        return false;
    if (pl_color_space_is_hdr(&src->color)) {
        if (!d.hdr_warned) {
            qCInfo(chiakiGui) << "[neural-render] HDR stream detected, DLSS neural rendering is skipped for HDR content";
            d.hdr_warned = true;
        }
        return false;
    }
    const int w = static_cast<int>(std::lround(std::fabs(pl_rect_w(src->crop))));
    const int h = static_cast<int>(std::lround(std::fabs(pl_rect_h(src->crop))));
    if (w <= 0 || h <= 0)
        return false;
    const uint64_t key = signature ^ (static_cast<uint64_t>(w) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(h) << 32)
        ^ 0x4E52000000000000ull;

    auto fill_frame = [&](Impl::CacheEntry &entry) {
        entry.last_use = ++d.use_counter;
        *out_frame = {};
        out_frame->num_planes = 1;
        out_frame->planes[0].texture = entry.image.tex;
        out_frame->planes[0].components = 3;
        out_frame->planes[0].component_mapping[0] = PL_CHANNEL_R;
        out_frame->planes[0].component_mapping[1] = PL_CHANNEL_G;
        out_frame->planes[0].component_mapping[2] = PL_CHANNEL_B;
        out_frame->repr = pl_color_repr_rgb;
        out_frame->color = pl_color_space_bt709;
        out_frame->crop = { 0.0f, 0.0f, static_cast<float>(entry.image.w), static_cast<float>(entry.image.h) };
        *out_signature = entry.key;
    };
    for (auto &entry : d.cache) {
        if (entry.key == key && entry.image.tex) {
            fill_frame(entry);
            return true;
        }
    }

    // Resizing destroys the texture, so pending copies must not use it anymore.
    if (d.input.tex && (d.input.w != w || d.input.h != h) && !d.waitDone(d.done_value))
        return false;
    if (!d.ensureImage(d.input, w, h))
        return false;
    struct pl_frame in_target = {};
    in_target.num_planes = 1;
    in_target.planes[0].texture = d.input.tex;
    in_target.planes[0].components = 4;
    for (int i = 0; i < 4; i++)
        in_target.planes[0].component_mapping[i] = i;
    in_target.repr = pl_color_repr_rgb;
    in_target.color = pl_color_space_bt709;
    struct pl_render_params render_params = pl_render_default_params;
    render_params.frame_mixer = nullptr;
    if (!pl_render_image(d.renderer, src, &in_target, &render_params)) {
        qCWarning(chiakiGui) << "[neural-render] Failed to prepare the input frame";
        return false;
    }

    Impl::CacheEntry &entry = d.acquireCacheEntry(key);
    // A recycled entry is several frames old, so this wait normally returns at once.
    if (!d.waitDone(entry.done_value) || !d.ensureImage(entry.image, w, h)) {
        entry.key = 0;
        return false;
    }
    bool crashed = false;
    const bool ok = d.run(entry.image, w, h, d.reset_pending, &crashed);
    if (!ok) {
        entry.key = 0;
        d.reset_pending = true;
        if (crashed || ++d.failures >= kMaxFailures) {
            qCWarning(chiakiGui) << "[neural-render] Disabling DLSS neural rendering for this session"
                                 << (crashed ? "(its runtime crashed)" : "(repeated failures)");
            if (crashed)
                d.abandonRunner();
            d.shutdown();
            d.init_failed = true;
        }
        return false;
    }
    d.failures = 0;
    d.reset_pending = false;
    entry.done_value = d.done_value;
    fill_frame(entry);
    return true;
}

#else // !CHIAKI_HAVE_DX12FG

struct NeuralRender::Impl {};

bool NeuralRender::compiledIn() { return false; }
NeuralRender::NeuralRender(pl_log) {}
NeuralRender::~NeuralRender() = default;
bool NeuralRender::ensureInitialized(pl_vulkan) { return false; }
bool NeuralRender::failed() const { return true; }
float NeuralRender::gpuTimeMs() const { return 0.0f; }
void NeuralRender::destroy() {}
bool NeuralRender::process(const struct pl_frame *, uint64_t, struct pl_frame *, uint64_t *) { return false; }

#endif // CHIAKI_HAVE_DX12FG
