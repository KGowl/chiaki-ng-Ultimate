#include "dx12fg.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winver.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <vulkan/vulkan_win32.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_dlssg_d3d.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr size_t kSlots = 4;
constexpr UINT kHiddenWindowSize = 1280;
// A project id of our own, as NGX wants one per application.
constexpr const char *kProjectId = "9d1c7a52-6a4e-4c36-8d4f-1f0a5f3b7c11";
constexpr UINT kVendorNvidia = 0x10DE;
// DLSS Neural Rendering's NGX feature id (a reserved slot in the public headers).
constexpr int kFeatureNeuralRendering = 18;
// NGX's generic application id, for the snippet's own Init_Ext.
constexpr unsigned long long kNrApplicationId = 0x24480451ull;

// NVIDIA's DLSS Neural Rendering snippet (nvngx_dlssnr.dll) only serves callers inside a
// module named like nvngx.dll; nvngx.dll_dlssnr.dll (src/nrgate.cpp) is that pass-through.
struct NrGate
{
    int (*load)(const wchar_t *) = nullptr;
    int (*init)(unsigned long long, const wchar_t *, void *, int, const void *) = nullptr;
    int (*create)(void *, int, void *, void **) = nullptr;
    int (*evaluate)(void *, void *, void *) = nullptr;
    int (*release)(void *) = nullptr;
    int (*shutdown)(void *) = nullptr;
    const char *(*fault)() = nullptr;

    bool open(const std::wstring &path)
    {
        HMODULE m = LoadLibraryW(path.c_str());
        if (!m)
            return false;
        load = reinterpret_cast<decltype(load)>(GetProcAddress(m, "nrgate_load"));
        init = reinterpret_cast<decltype(init)>(GetProcAddress(m, "nrgate_init"));
        create = reinterpret_cast<decltype(create)>(GetProcAddress(m, "nrgate_create"));
        evaluate = reinterpret_cast<decltype(evaluate)>(GetProcAddress(m, "nrgate_evaluate"));
        release = reinterpret_cast<decltype(release)>(GetProcAddress(m, "nrgate_release"));
        shutdown = reinterpret_cast<decltype(shutdown)>(GetProcAddress(m, "nrgate_shutdown"));
        fault = reinterpret_cast<decltype(fault)>(GetProcAddress(m, "nrgate_fault"));
        return load && init && create && evaluate && release && shutdown && fault;
    }
};

std::string fileVersion(const std::wstring &path)
{
    DWORD unused = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &unused);
    if (!size)
        return "unknown version";
    std::vector<char> data(size);
    VS_FIXEDFILEINFO *info = nullptr;
    UINT len = 0;
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data())
        || !VerQueryValueW(data.data(), L"\\", reinterpret_cast<void **>(&info), &len) || !info)
        return "unknown version";
    char text[64];
    std::snprintf(text, sizeof(text), "%u.%u.%u.%u", HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
                  HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS));
    return text;
}

const char *ngxResultString(NVSDK_NGX_Result r)
{
    switch (r) {
    case NVSDK_NGX_Result_Success: return "Success";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError: return "PlatformError";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FeatureNotFound";
    case NVSDK_NGX_Result_FAIL_InvalidParameter: return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_OutOfDate: return "OutOfDate";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "UnableToInitializeFeature";
    default: return "Fail";
    }
}

// DLSS Super Resolution's quality mode for an upscale from in_w to out_w (DLAA at 1:1).
NVSDK_NGX_PerfQuality_Value qualityFor(int in_w, int out_w)
{
    const double s = static_cast<double>(out_w) / static_cast<double>(in_w);
    return s < 1.01   ? NVSDK_NGX_PerfQuality_Value_DLAA
           : s < 1.6  ? NVSDK_NGX_PerfQuality_Value_MaxQuality
           : s < 1.85 ? NVSDK_NGX_PerfQuality_Value_Balanced
           : s < 2.5  ? NVSDK_NGX_PerfQuality_Value_MaxPerf
                      : NVSDK_NGX_PerfQuality_Value_UltraPerformance;
}

const char *qualityName(NVSDK_NGX_PerfQuality_Value q)
{
    switch (q) {
    case NVSDK_NGX_PerfQuality_Value_DLAA: return "DLAA";
    case NVSDK_NGX_PerfQuality_Value_MaxQuality: return "Quality";
    case NVSDK_NGX_PerfQuality_Value_Balanced: return "Balanced";
    case NVSDK_NGX_PerfQuality_Value_MaxPerf: return "Performance";
    case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return "Ultra Performance";
    default: return "?";
    }
}

// HDR through feature 18, which only takes SDR (it clips at 1.0): the model sees the proxy
// p = C / (1 + P), P the brightest channel of the linear pixel C (so p stays below 1), sRGB
// coded; its change to p is scaled back by (1 + P) onto C, so an unchanged proxy leaves the
// pixel as it was and the highlights stay HDR.
const char kProxyShaders[] = R"(
Texture2D<float4> src0 : register(t0);
Texture2D<float4> src1 : register(t1);
Texture2D<float4> src2 : register(t2);
RWTexture2D<float4> dst : register(u0);
cbuffer Constants : register(b0) { uint2 size; float max_out; float unused; };
float3 toSrgb(float3 c)
{
    c = saturate(c);
    return lerp(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, step(0.0031308, c));
}
float3 fromSrgb(float3 c)
{
    return lerp(c / 12.92, pow((c + 0.055) / 1.055, 2.4), step(0.04045, c));
}
[numthreads(8, 8, 1)]
void encode(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
        return;
    float3 c = max(src0[id.xy].rgb, 0.0);
    float m = max(c.r, max(c.g, c.b));
    dst[id.xy] = float4(toSrgb(c / (1.0 + m)), 1.0);
}
[numthreads(8, 8, 1)]
void compose(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
        return;
    float3 c = src0[id.xy].rgb;
    float m = max(max(c.r, max(c.g, c.b)), 0.0);
    float3 delta = fromSrgb(src2[id.xy].rgb) - fromSrgb(src1[id.xy].rgb);
    float3 o = max(c + delta * (1.0 + m), min(c, 0.0));
    dst[id.xy] = float4(min(o, max_out), 1.0);
}
)";

LRESULT CALLBACK hiddenWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// A stationary virtual camera: video carries no camera, but the projection matrices and
// the camera basis still have to agree (same values as the Vulkan DLSS backend).
void fillCamera(NVSDK_NGX_DLSSG_Opt_Eval_Params &opt, int w, int h)
{
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            const float v = r == c ? 1.0f : 0.0f;
            opt.cameraViewToClip[r][c] = v;
            opt.clipToCameraView[r][c] = v;
            opt.clipToLensClip[r][c] = v;
            opt.clipToPrevClip[r][c] = v;
            opt.prevClipToClip[r][c] = v;
        }
    }
    opt.cameraNear = 0.1f;
    opt.cameraFar = 1000.0f;
    opt.cameraFOV = 1.5707963f; // pi/2, so tan(fov / 2) = 1
    opt.cameraAspectRatio = static_cast<float>(w) / static_cast<float>(h);
    opt.cameraUp[1] = 1.0f;
    opt.cameraRight[0] = 1.0f;
    opt.cameraFwd[2] = 1.0f;
    const float proj_x = 1.0f / opt.cameraAspectRatio;
    const float proj_z = opt.cameraFar / (opt.cameraFar - opt.cameraNear);
    const float proj_w = -opt.cameraNear * proj_z;
    opt.cameraViewToClip[0][0] = proj_x;
    opt.cameraViewToClip[2][2] = proj_z;
    opt.cameraViewToClip[2][3] = 1.0f;
    opt.cameraViewToClip[3][2] = proj_w;
    opt.cameraViewToClip[3][3] = 0.0f;
    opt.clipToCameraView[0][0] = 1.0f / proj_x;
    opt.clipToCameraView[2][2] = 0.0f;
    opt.clipToCameraView[2][3] = 1.0f / proj_w;
    opt.clipToCameraView[3][2] = 1.0f;
    opt.clipToCameraView[3][3] = -proj_z / proj_w;
}

} // namespace

struct Dx12Fg::Impl
{
    VulkanDevice vk;
    LogFn log_fn = nullptr;
    void *log_user = nullptr;
    const char *stage = "";

    void log(int level, const char *fmt, ...)
    {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (log_fn)
            log_fn(level, buf, log_user);
    }

    // ---- Vulkan entry points ----
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR = nullptr;
    PFN_vkCreateSemaphore vkCreateSemaphore = nullptr;
    PFN_vkDestroySemaphore vkDestroySemaphore = nullptr;
    PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR = nullptr;
    PFN_vkSignalSemaphore vkSignalSemaphore = nullptr;
    PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = nullptr;
    PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage = nullptr;
    VkPhysicalDeviceMemoryProperties memory_props = {};
    LUID luid = {};

    // ---- D3D12 ----
    HMODULE dxgi_dll = nullptr;
    HMODULE d3d12_dll = nullptr;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    // The fence shared with Vulkan: its values are the timeline of the two sides.
    ComPtr<ID3D12Fence> shared_fence;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    uint64_t last_value = 0;
    // Private fence for the host to wait on (initialisation, resizing, command list reuse).
    ComPtr<ID3D12Fence> idle_fence;
    uint64_t idle_value = 0;
    HANDLE idle_event = nullptr;
    struct Slot
    {
        ComPtr<ID3D12CommandAllocator> allocator;
        uint64_t done_value = 0;
    };
    std::array<Slot, kSlots> slots;
    size_t next_slot = 0;
    ComPtr<ID3D12GraphicsCommandList> list;
    HWND hwnd = nullptr;
    ATOM window_class = 0;
    ComPtr<IDXGISwapChain1> swapchain;

    // ---- NGX ----
    bool ngx_up = false;
    bool fg_available = false;
    bool sr_available = false;
    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *handle = nullptr;
    int max_generated = 1;
    unsigned long long frame_id = 0;

    // ---- size dependent ----
    struct Tex
    {
        ComPtr<ID3D12Resource> res;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        UINT bytes_per_texel = 4;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT64 total = 0;
    };
    struct Shared
    {
        ComPtr<ID3D12Resource> res;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        UINT64 size = 0;
    };
    int width = 0;
    int height = 0;
    Tex color, motion, depth;
    std::array<Tex, kMaxOutputs> outs;
    Shared color_buf, motion_buf;
    std::array<Shared, kMaxOutputs> out_bufs;
    bool sized = false;

    // ---- neural rendering, own size: NGX feature 18 through the gate on NVIDIA, DLSS Super
    // Resolution in DLAA mode elsewhere (NeuralHub runs DLSS 5 in front of it) ----
    UINT adapter_vendor = 0;
    std::wstring exe_dir;
    std::wstring ngx_data;
    NrGate gate;
    bool gate_ready = false;
    NVSDK_NGX_Parameter *nr_params = nullptr;
    NVSDK_NGX_Handle *nr_handle = nullptr;
    // NVIDIA upscaling: DLSS Super Resolution from feature 18's output.
    NVSDK_NGX_Parameter *nr_sr_params = nullptr;
    NVSDK_NGX_Handle *nr_sr_handle = nullptr;
    NrConfig nr_cfg;
    Tex nr_in, nr_out, nr_depth, nr_motion;
    // NVIDIA: feature 18's output at the input size when upscaling follows (nr_mid), and
    // the RGBA8 SDR proxy of an HDR frame before / after the model.
    Tex nr_mid, nr_proxy_in, nr_proxy_out;
    Shared nr_in_buf, nr_out_buf;
    bool nr_sized = false;
    // HDR proxy compute passes (NVIDIA): encode = linear frame -> proxy, compose = linear
    // frame + (model's proxy - proxy) -> output. Shaders compiled once with D3DCompile.
    ComPtr<ID3D12RootSignature> cs_root;
    ComPtr<ID3D12PipelineState> cs_encode, cs_compose;
    ComPtr<ID3D12DescriptorHeap> cs_heap;
    UINT cs_increment = 0;
    bool cs_failed = false;
    // GPU time of the evaluation: two timestamps resolved into a readback buffer, read
    // once the command list that wrote them has finished.
    ComPtr<ID3D12QueryHeap> nr_queries;
    ComPtr<ID3D12Resource> nr_query_readback;
    UINT64 timestamp_frequency = 0;
    uint64_t nr_query_done = 0;
    double nr_gpu_ms = 0.0;
    // The runtime crashed while one of the users called it; never handed out again.
    bool crashed = false;

    bool loadVulkan();
    bool signalShared(uint64_t value);
    bool createDevice();
    bool createSwapchain();
    bool initNgx();
    void initGate();
    bool createSharedFence();
    bool makeTex(Tex &t, int w, int h, DXGI_FORMAT format, UINT bytes_per_texel);
    bool fill(const std::vector<Tex *> &zero, Tex *half);
    bool nrCreateFeature();
    bool nrCreateUpscaler();
    bool nrEvaluate(bool reset, int log_level = 1);
    bool nrWarmUp();
    void destroyNr();
    // Every neural rendering texture that exists, for warm-up fills and state rollback.
    std::vector<Tex *> nrTextures()
    {
        std::vector<Tex *> all;
        for (Tex *t : { &nr_in, &nr_out, &nr_depth, &nr_motion, &nr_mid, &nr_proxy_in, &nr_proxy_out }) {
            if (t->res)
                all.push_back(t);
        }
        return all;
    }
    bool nrUsesProxy() const { return gate_ready && nr_cfg.hdr; }
    bool nrUpscales() const { return nr_cfg.out_width != nr_cfg.width || nr_cfg.out_height != nr_cfg.height; }
    bool createProxyPipeline();
    void writeProxyDescriptors();
    void dispatchProxy(bool compose);
    bool makeShared(Shared &s, UINT64 size);
    void freeShared(Shared &s);
    bool waitIdle();
    bool beginList(Slot **out_slot);
    bool endList(Slot &slot);
    void barrier(Tex &t, D3D12_RESOURCE_STATES to);
    void copyBufferToTex(Tex &t, ID3D12Resource *buffer);
    void copyTexToBuffer(ID3D12Resource *buffer, Tex &t);
    bool createFeature();
    void releaseFeature();
    bool evaluate(int count, bool reset, float msx, float msy, int log_level = 1);
    bool warmUp();
    struct States
    {
        D3D12_RESOURCE_STATES color, motion, depth;
        std::array<D3D12_RESOURCE_STATES, kMaxOutputs> outs;
    };
    States saveStates() const;
    void restoreStates(const States &s);
    void destroySized();
};

bool Dx12Fg::Impl::loadVulkan()
{
    bool ok = true;
#define DX12FG_VK(name_) \
    name_ = reinterpret_cast<PFN_##name_>(vk.get_instance_proc_addr(vk.instance, #name_)); \
    if (!name_) { log(1, "the Vulkan device lacks %s", #name_); ok = false; }
    DX12FG_VK(vkGetPhysicalDeviceProperties2)
    DX12FG_VK(vkGetPhysicalDeviceMemoryProperties)
    DX12FG_VK(vkCreateBuffer)
    DX12FG_VK(vkDestroyBuffer)
    DX12FG_VK(vkGetBufferMemoryRequirements)
    DX12FG_VK(vkAllocateMemory)
    DX12FG_VK(vkFreeMemory)
    DX12FG_VK(vkBindBufferMemory)
    DX12FG_VK(vkGetMemoryWin32HandlePropertiesKHR)
    DX12FG_VK(vkCreateSemaphore)
    DX12FG_VK(vkDestroySemaphore)
    DX12FG_VK(vkImportSemaphoreWin32HandleKHR)
    DX12FG_VK(vkSignalSemaphore)
    DX12FG_VK(vkCmdCopyImageToBuffer)
    DX12FG_VK(vkCmdCopyBufferToImage)
#undef DX12FG_VK
    if (!ok) {
        log(1, "the Vulkan device cannot share memory with D3D12 (VK_KHR_external_memory_win32 and "
               "VK_KHR_external_semaphore_win32 missing?)");
        return false;
    }
    vkGetPhysicalDeviceMemoryProperties(vk.physical_device, &memory_props);
    VkPhysicalDeviceIDProperties id_props = {};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props = {};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(vk.physical_device, &props);
    if (!id_props.deviceLUIDValid) {
        log(1, "the Vulkan device has no LUID, cannot find the matching D3D12 adapter");
        return false;
    }
    std::memcpy(&luid, id_props.deviceLUID, sizeof(luid));
    return true;
}

bool Dx12Fg::Impl::createDevice()
{
    // A dxgi.dll next to the executable is the NGX provider's stub (NeuralHub). Windows
    // would load it before the system one, but only if nothing loaded the system dxgi.dll
    // first (Vulkan drivers do), so load it by its full path; it forwards to the system one.
    {
        wchar_t exe[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring proxy = exe;
        proxy = proxy.substr(0, proxy.find_last_of(L'\\') + 1) + L"dxgi.dll";
        if (GetFileAttributesW(proxy.c_str()) != INVALID_FILE_ATTRIBUTES) {
            dxgi_dll = LoadLibraryW(proxy.c_str());
            if (dxgi_dll)
                log(0, "using the dxgi.dll next to the executable");
        }
    }
    if (!dxgi_dll)
        dxgi_dll = LoadLibraryW(L"dxgi.dll");
    d3d12_dll = LoadLibraryW(L"d3d12.dll");
    if (!dxgi_dll || !d3d12_dll) {
        log(1, "dxgi.dll / d3d12.dll could not be loaded");
        return false;
    }
    using CreateFactoryFn = HRESULT(WINAPI *)(REFIID, void **);
    auto create_factory = reinterpret_cast<CreateFactoryFn>(GetProcAddress(dxgi_dll, "CreateDXGIFactory1"));
    auto create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(d3d12_dll, "D3D12CreateDevice"));
    if (!create_factory || !create_device) {
        log(1, "CreateDXGIFactory1 / D3D12CreateDevice not found");
        return false;
    }
    if (FAILED(create_factory(IID_PPV_ARGS(&factory)))) {
        log(1, "CreateDXGIFactory1 failed");
        return false;
    }
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
        log(1, "no D3D12 adapter matches the Vulkan GPU");
        return false;
    }
    DXGI_ADAPTER_DESC1 desc = {};
    adapter->GetDesc1(&desc);
    char name[128] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
    adapter_vendor = desc.VendorId;
    HRESULT hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        log(1, "D3D12CreateDevice failed on %s (0x%08x)", name, static_cast<unsigned>(hr));
        return false;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))))
        return false;
    for (auto &slot : slots) {
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.allocator))))
            return false;
    }
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, slots[0].allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&list))))
        return false;
    list->Close();
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&idle_fence))))
        return false;
    idle_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    log(0, "D3D12 device on %s", name);
    return idle_event != nullptr;
}

bool Dx12Fg::Impl::waitIdle()
{
    const uint64_t value = ++idle_value;
    if (FAILED(queue->Signal(idle_fence.Get(), value)))
        return false;
    if (idle_fence->GetCompletedValue() < value) {
        if (FAILED(idle_fence->SetEventOnCompletion(value, idle_event)))
            return false;
        if (WaitForSingleObject(idle_event, 15000) != WAIT_OBJECT_0) {
            log(1, "the D3D12 GPU work did not finish within 15 s");
            return false;
        }
    }
    return true;
}

bool Dx12Fg::Impl::createSwapchain()
{
    // The NGX D3D12 runtime of NeuralHub opens at the first Present of a swapchain on
    // the device; without one the evaluation fails. A hidden window is enough. The
    // provider installs its hooks on the swapchain calls a moment after the first
    // device is created, so a swapchain made too early goes unnoticed: this is called
    // again (with a new swapchain on the same window) when the runtime stays closed.
    swapchain.Reset();
    WNDCLASSW wc = {};
    wc.lpfnWndProc = hiddenWndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"chiaki_dx12fg_hidden_window";
    if (!hwnd) {
        window_class = RegisterClassW(&wc);
        if (!window_class && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            log(1, "could not register the hidden window class");
            return false;
        }
        hwnd = CreateWindowExW(0, wc.lpszClassName, L"chiaki D3D12 frame generation", WS_OVERLAPPEDWINDOW, 0, 0,
                               kHiddenWindowSize, kHiddenWindowSize * 9 / 16, nullptr, nullptr, wc.hInstance, nullptr);
        if (!hwnd) {
            log(1, "could not create the hidden window");
            return false;
        }
    }
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = kHiddenWindowSize;
    sd.Height = kHiddenWindowSize * 9 / 16;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 3;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGIFactory2> factory2;
    if (FAILED(factory.As(&factory2)))
        return false;
    HRESULT hr = factory2->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &swapchain);
    if (FAILED(hr)) {
        log(1, "CreateSwapChainForHwnd failed (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
#if defined(CHIAKI_DX12FG_TEST_HOOKS)
    // Test hook (dev builds only): the first swapchain is never presented, as if the provider
    // had missed it, to exercise the retry in warmUp().
    static bool skipped_first = false;
    if (!skipped_first && std::getenv("CHIAKI_DX12FG_SKIP_FIRST_PRESENT")) {
        skipped_first = true;
        log(0, "test hook: not presenting the first swapchain");
        return true;
    }
#endif
    hr = swapchain->Present(0, 0);
    if (FAILED(hr)) {
        log(1, "the first Present failed (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    return waitIdle();
}

bool Dx12Fg::Impl::initNgx()
{
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe;
    dir = dir.substr(0, dir.find_last_of(L'\\'));
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    const std::wstring data = std::wstring(tmp) + L"chiaki-ng-ngx-d3d12";
    CreateDirectoryW(data.c_str(), nullptr);
    exe_dir = dir;
    ngx_data = data;
    const wchar_t *paths[] = { dir.c_str() };
    NVSDK_NGX_FeatureCommonInfo info = {};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    stage = "NVSDK_NGX_D3D12_Init";
    NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", data.c_str(),
                                                             device.Get(), &info, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        log(1, "NGX D3D12 init failed: %s (0x%08x). On a GPU without NVIDIA's runtime this needs a NGX provider "
               "next to chiaki.exe (NeuralHub's dxgi.dll stub and neuralhub.ini)",
            ngxResultString(r), static_cast<unsigned>(r));
        return false;
    }
    ngx_up = true;
    NVSDK_NGX_Parameter *caps = nullptr;
    stage = "NVSDK_NGX_D3D12_GetCapabilityParameters";
    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&caps);
    if (NVSDK_NGX_FAILED(r) || !caps) {
        log(1, "NGX D3D12 capability query failed: %s", ngxResultString(r));
        return false;
    }
    auto capability = [caps](const char *name) {
        unsigned int value = 0;
        if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetUI(caps, name, &value))) {
            int value_i = 0;
            NVSDK_NGX_Parameter_GetI(caps, name, &value_i);
            value = value_i > 0 ? 1 : 0;
        }
        return value != 0;
    };
    fg_available = capability(NVSDK_NGX_Parameter_FrameGeneration_Available);
    sr_available = capability(NVSDK_NGX_Parameter_SuperSampling_Available);
    unsigned int multi_frame_max = 0;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &multi_frame_max))
        && multi_frame_max > 0)
        max_generated = std::min(static_cast<int>(multi_frame_max), kMaxOutputs);
    NVSDK_NGX_D3D12_DestroyParameters(caps);
    stage = "NVSDK_NGX_D3D12_AllocateParameters";
    r = NVSDK_NGX_D3D12_AllocateParameters(&params);
    if (NVSDK_NGX_SUCCEED(r) && params)
        r = NVSDK_NGX_D3D12_AllocateParameters(&nr_params);
    if (NVSDK_NGX_FAILED(r) || !params || !nr_params) {
        log(1, "NGX D3D12 parameter allocation failed: %s", ngxResultString(r));
        return false;
    }
    if (fg_available)
        log(0, "DLSS Frame Generation (D3D12) available, up to %d generated frames", max_generated);
    else
        log(1, "DLSS Frame Generation is not available through NGX D3D12 on this GPU");
    if (adapter_vendor == kVendorNvidia) {
        initGate();
        log(0, "DLSS Super Resolution (upscaling after neural rendering) %s",
            sr_available ? "available" : "not available (needs nvngx_dlss.dll next to chiaki.exe)");
    } else {
        log(sr_available ? 0 : 1, "DLSS Super Resolution (D3D12, used for neural rendering) %s",
            sr_available ? "available" : "is not available");
    }
    return true;
}

// NVIDIA: DLSS 5 Neural Rendering (NGX feature 18) straight from nvngx_dlssnr.dll next to
// chiaki.exe, called through nvngx.dll_dlssnr.dll. The driver core is never asked for
// feature 18 (from driver 616.64 on it routes that request into the snippet and faults).
// Not fatal: without it there is no neural rendering, frame generation is unaffected.
void Dx12Fg::Impl::initGate()
{
    const std::wstring snippet = exe_dir + L"\\nvngx_dlssnr.dll";
    if (GetFileAttributesW(snippet.c_str()) == INVALID_FILE_ATTRIBUTES) {
        log(1, "DLSS 5 neural rendering needs NVIDIA's nvngx_dlssnr.dll next to chiaki.exe (not bundled)");
        return;
    }
    stage = "neural rendering gate";
    if (!gate.open(exe_dir + L"\\nvngx.dll_dlssnr.dll")) {
        log(1, "nvngx.dll_dlssnr.dll (part of chiaki-ng) is missing or incomplete next to chiaki.exe");
        return;
    }
    if (!gate.load(snippet.c_str())) {
        log(1, "nvngx_dlssnr.dll could not be loaded: %s", gate.fault());
        return;
    }
    stage = "nvngx_dlssnr Init_Ext";
    const auto r = static_cast<NVSDK_NGX_Result>(
        gate.init(kNrApplicationId, ngx_data.c_str(), device.Get(), NVSDK_NGX_Version_API, nr_params));
    if (NVSDK_NGX_FAILED(r)) {
        log(1, "nvngx_dlssnr.dll refused to initialise: %s (0x%08x) %s", ngxResultString(r), static_cast<unsigned>(r),
            gate.fault());
        return;
    }
    gate_ready = true;
    const std::string version = fileVersion(snippet);
    log(0, "DLSS 5 neural rendering (NGX feature 18) ready, nvngx_dlssnr.dll %s", version.c_str());
}

bool Dx12Fg::Impl::createSharedFence()
{
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared_fence));
    if (FAILED(hr)) {
        log(1, "could not create the shared D3D12 fence (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    HANDLE handle_nt = nullptr;
    hr = device->CreateSharedHandle(shared_fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle_nt);
    if (FAILED(hr) || !handle_nt) {
        log(1, "could not share the D3D12 fence (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    VkSemaphoreTypeCreateInfo type_info = {};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sem_info = {};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &type_info;
    bool ok = vkCreateSemaphore(vk.device, &sem_info, nullptr, &semaphore) == VK_SUCCESS;
    if (ok) {
        VkImportSemaphoreWin32HandleInfoKHR import_info = {};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
        import_info.semaphore = semaphore;
        import_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
        import_info.handle = handle_nt;
        ok = vkImportSemaphoreWin32HandleKHR(vk.device, &import_info) == VK_SUCCESS;
    }
    // Importing does not take over the handle.
    CloseHandle(handle_nt);
    if (!ok)
        log(1, "Vulkan could not import the D3D12 fence as a timeline semaphore");
    return ok;
}

bool Dx12Fg::Impl::makeTex(Tex &t, int w, int h, DXGI_FORMAT format, UINT bytes_per_texel)
{
    t.format = format;
    t.bytes_per_texel = bytes_per_texel;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = static_cast<UINT64>(w);
    rd.Height = static_cast<UINT>(h);
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = format;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&t.res));
    if (FAILED(hr)) {
        log(1, "could not create a %dx%d D3D12 texture (0x%08x)", w, h, static_cast<unsigned>(hr));
        return false;
    }
    t.state = D3D12_RESOURCE_STATE_COMMON;
    UINT rows = 0;
    UINT64 row_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &t.footprint, &rows, &row_size, &t.total);
    return true;
}

bool Dx12Fg::Impl::makeShared(Shared &s, UINT64 size)
{
    s.size = size;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&s.res));
    if (FAILED(hr)) {
        log(1, "could not create a shared D3D12 buffer (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    HANDLE handle_nt = nullptr;
    hr = device->CreateSharedHandle(s.res.Get(), nullptr, GENERIC_ALL, nullptr, &handle_nt);
    if (FAILED(hr) || !handle_nt) {
        log(1, "could not share a D3D12 buffer (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    bool ok = false;
    do {
        VkExternalMemoryBufferCreateInfo ext_info = {};
        ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        VkBufferCreateInfo buf_info = {};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.pNext = &ext_info;
        buf_info.size = size;
        buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(vk.device, &buf_info, nullptr, &s.buffer) != VK_SUCCESS)
            break;
        VkMemoryRequirements reqs = {};
        vkGetBufferMemoryRequirements(vk.device, s.buffer, &reqs);
        uint32_t bits = reqs.memoryTypeBits;
        VkMemoryWin32HandlePropertiesKHR handle_props = {};
        handle_props.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR;
        if (vkGetMemoryWin32HandlePropertiesKHR(vk.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, handle_nt,
                                                &handle_props) == VK_SUCCESS && handle_props.memoryTypeBits)
            bits &= handle_props.memoryTypeBits;
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory_props.memoryTypeCount && type == UINT32_MAX; i++) {
            if ((bits & (1u << i)) && (memory_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                type = i;
        }
        for (uint32_t i = 0; i < memory_props.memoryTypeCount && type == UINT32_MAX; i++) {
            if (bits & (1u << i))
                type = i;
        }
        if (type == UINT32_MAX)
            break;
        VkMemoryDedicatedAllocateInfo dedicated = {};
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.buffer = s.buffer;
        VkImportMemoryWin32HandleInfoKHR import_info = {};
        import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        import_info.pNext = &dedicated;
        import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        import_info.handle = handle_nt;
        VkMemoryAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.pNext = &import_info;
        alloc.allocationSize = reqs.size;
        alloc.memoryTypeIndex = type;
        if (vkAllocateMemory(vk.device, &alloc, nullptr, &s.memory) != VK_SUCCESS)
            break;
        if (vkBindBufferMemory(vk.device, s.buffer, s.memory, 0) != VK_SUCCESS)
            break;
        ok = true;
    } while (false);
    // Importing does not take over the handle.
    CloseHandle(handle_nt);
    if (!ok)
        log(1, "Vulkan could not import a shared D3D12 buffer");
    return ok;
}

void Dx12Fg::Impl::freeShared(Shared &s)
{
    if (s.buffer)
        vkDestroyBuffer(vk.device, s.buffer, nullptr);
    if (s.memory)
        vkFreeMemory(vk.device, s.memory, nullptr);
    s.res.Reset();
    s = Shared();
}

bool Dx12Fg::Impl::beginList(Slot **out_slot)
{
    Slot &slot = slots[next_slot];
    next_slot = (next_slot + 1) % slots.size();
    if (slot.done_value && idle_fence->GetCompletedValue() < slot.done_value) {
        if (FAILED(idle_fence->SetEventOnCompletion(slot.done_value, idle_event)) ||
            WaitForSingleObject(idle_event, 15000) != WAIT_OBJECT_0) {
            log(1, "a D3D12 command allocator stayed busy for 15 s");
            return false;
        }
    }
    if (FAILED(slot.allocator->Reset()) || FAILED(list->Reset(slot.allocator.Get(), nullptr)))
        return false;
    *out_slot = &slot;
    return true;
}

bool Dx12Fg::Impl::endList(Slot &slot)
{
    if (FAILED(list->Close()))
        return false;
    ID3D12CommandList *lists[] = { list.Get() };
    queue->ExecuteCommandLists(1, lists);
    slot.done_value = ++idle_value;
    return SUCCEEDED(queue->Signal(idle_fence.Get(), slot.done_value));
}

void Dx12Fg::Impl::barrier(Tex &t, D3D12_RESOURCE_STATES to)
{
    if (t.state == to)
        return;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = t.res.Get();
    b.Transition.StateBefore = t.state;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &b);
    t.state = to;
}

void Dx12Fg::Impl::copyBufferToTex(Tex &t, ID3D12Resource *buffer)
{
    barrier(t, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = t.res.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = buffer;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = t.footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

void Dx12Fg::Impl::copyTexToBuffer(ID3D12Resource *buffer, Tex &t)
{
    barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    src.pResource = t.res.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = t.footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

bool Dx12Fg::Impl::createFeature()
{
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_DLSSG_Parameter_ResourceAlwaysProvided_Flags,
        NVSDK_NGX_DLSSG_ResourceFlags_Backbuffer | NVSDK_NGX_DLSSG_ResourceFlags_MVecs
        | NVSDK_NGX_DLSSG_ResourceFlags_Depth | NVSDK_NGX_DLSSG_ResourceFlags_OutputInterpolated);
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_DLSSG_Parameter_ResourceNeverProvided_Flags,
        NVSDK_NGX_DLSSG_ResourceFlags_HUDLess | NVSDK_NGX_DLSSG_ResourceFlags_UI
        | NVSDK_NGX_DLSSG_ResourceFlags_UIAlpha | NVSDK_NGX_DLSSG_ResourceFlags_BidirectionalDistortionField);
    NVSDK_NGX_DLSSG_Create_Params cp = {};
    cp.Width = static_cast<unsigned int>(width);
    cp.Height = static_cast<unsigned int>(height);
    cp.NativeBackbufferFormat = static_cast<unsigned int>(DXGI_FORMAT_R8G8B8A8_UNORM);
    cp.RenderWidth = cp.Width;
    cp.RenderHeight = cp.Height;
    cp.DynamicResolutionScaling = false;
    Slot *slot = nullptr;
    if (!beginList(&slot))
        return false;
    stage = "NGX_D3D12_CREATE_DLSSG";
    const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSSG(list.Get(), 1, 1, &handle, params, &cp);
    const bool executed = endList(*slot) && waitIdle();
    if (NVSDK_NGX_FAILED(r) || !handle) {
        log(1, "creating DLSS Frame Generation (D3D12) failed: %s (0x%08x)", ngxResultString(r), static_cast<unsigned>(r));
        handle = nullptr;
        return false;
    }
    return executed;
}

void Dx12Fg::Impl::releaseFeature()
{
    if (handle)
        NVSDK_NGX_D3D12_ReleaseFeature(handle);
    handle = nullptr;
}

// Records the evaluations for the current frame into `list`; the colour and motion
// textures hold the current frame.
Dx12Fg::Impl::States Dx12Fg::Impl::saveStates() const
{
    States s = {};
    s.color = color.state;
    s.motion = motion.state;
    s.depth = depth.state;
    for (size_t i = 0; i < outs.size(); i++)
        s.outs[i] = outs[i].state;
    return s;
}

void Dx12Fg::Impl::restoreStates(const States &s)
{
    color.state = s.color;
    motion.state = s.motion;
    depth.state = s.depth;
    for (size_t i = 0; i < outs.size(); i++)
        outs[i].state = s.outs[i];
}

bool Dx12Fg::Impl::evaluate(int count, bool reset, float msx, float msy, int log_level)
{
    barrier(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    barrier(motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    barrier(depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    NVSDK_NGX_Parameter_SetULL(params, NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, ++frame_id);
    for (int i = 0; i < count; i++) {
        Tex &out = outs[static_cast<size_t>(i)];
        barrier(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        NVSDK_NGX_D3D12_DLSSG_Eval_Params eval = {};
        eval.pBackbuffer = color.res.Get();
        eval.pMVecs = motion.res.Get();
        eval.pDepth = depth.res.Get();
        eval.pOutputInterpFrame = out.res.Get();
        NVSDK_NGX_DLSSG_Opt_Eval_Params opt = {};
        // The count excludes the real frame; indices are one based.
        opt.multiFrameCount = static_cast<unsigned int>(count);
        opt.multiFrameIndex = static_cast<unsigned int>(i + 1);
        fillCamera(opt, width, height);
        opt.mvecScale[0] = msx;
        opt.mvecScale[1] = msy;
        opt.colorBuffersHDR = false;
        opt.depthInverted = false;
        opt.cameraMotionIncluded = true;
        opt.reset = reset;
        opt.motionVectorsInvalidValue = 3.402823466e38f;
        opt.minRelativeLinearDepthObjectSeparation = 40.0f;
        const NVSDK_NGX_Dimensions size = { static_cast<unsigned int>(width), static_cast<unsigned int>(height) };
        opt.backbufferSubrectSize = size;
        opt.outputInterpSubrectSize = size;
        opt.mvecsSubrectSize = size;
        opt.depthSubrectSize = size;
        stage = "NGX_D3D12_EVALUATE_DLSSG";
        const NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSSG(list.Get(), handle, params, &eval, &opt);
        if (NVSDK_NGX_FAILED(r)) {
            log(log_level, "evaluating DLSS Frame Generation (D3D12) failed: %s (0x%08x)", ngxResultString(r),
                static_cast<unsigned>(r));
            return false;
        }
    }
    return true;
}

// Uploads zeros into `zero` and a constant 0.5 into the R32_FLOAT texture `half`
// (a constant mid-range depth keeps disocclusion logic neutral), and waits for it.
bool Dx12Fg::Impl::fill(const std::vector<Tex *> &zero, Tex *half)
{
    UINT64 size = half ? half->total : 0;
    for (Tex *t : zero)
        size = std::max(size, t->total);
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload_zero;
    ComPtr<ID3D12Resource> upload_half;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&upload_zero))))
        return false;
    void *p = nullptr;
    if (FAILED(upload_zero->Map(0, nullptr, &p)))
        return false;
    std::memset(p, 0, static_cast<size_t>(size));
    upload_zero->Unmap(0, nullptr);
    if (half) {
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&upload_half))) ||
            FAILED(upload_half->Map(0, nullptr, &p)))
            return false;
        const auto &fp = half->footprint.Footprint;
        for (UINT y = 0; y < fp.Height; y++) {
            float *row = reinterpret_cast<float *>(static_cast<uint8_t *>(p) + static_cast<size_t>(y) * fp.RowPitch);
            for (UINT x = 0; x < fp.Width; x++)
                row[x] = 0.5f;
        }
        upload_half->Unmap(0, nullptr);
    }
    Slot *slot = nullptr;
    if (!beginList(&slot))
        return false;
    for (Tex *t : zero)
        copyBufferToTex(*t, upload_zero.Get());
    if (half)
        copyBufferToTex(*half, upload_half.Get());
    return endList(*slot) && waitIdle();
}

// The runtime builds its pipelines in the first two evaluations (about 0.2 s of CPU
// each): do it here, on dummy data, instead of during the stream.
bool Dx12Fg::Impl::warmUp()
{
    // Dummy content: zeroed colour and motion, constant depth.
    std::vector<Tex *> zero = { &color, &motion };
    for (auto &out : outs) {
        if (out.res)
            zero.push_back(&out);
    }
    if (!fill(zero, &depth))
        return false;

    Slot *slot = nullptr;
    for (int i = 0; i < 2; i++) {
        constexpr int kAttempts = 4;
        bool done = false;
        for (int attempt = 0; attempt < kAttempts && !done; attempt++) {
            if (attempt > 0) {
                // The provider's runtime is still closed: it did not see the first swapchain.
                log(0, "the NGX runtime is not open yet, presenting a new swapchain (attempt %d of %d)", attempt + 1,
                    kAttempts);
                Sleep(100u * static_cast<unsigned>(attempt));
                if (!createSwapchain())
                    return false;
            }
            const States saved = saveStates();
            if (!beginList(&slot))
                return false;
            // Quiet while a later attempt can still succeed.
            const bool ok = evaluate(1, true, 1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height),
                                     attempt + 1 < kAttempts ? 0 : 1);
            if (!ok) {
                // Nothing of this list runs: the textures keep the states they had.
                list->Close();
                restoreStates(saved);
                continue;
            }
            if (!endList(*slot) || !waitIdle())
                return false;
            done = true;
        }
        if (!done)
            return false;
    }
    return true;
}

void Dx12Fg::Impl::destroySized()
{
    if (queue)
        waitIdle();
    releaseFeature();
    freeShared(color_buf);
    freeShared(motion_buf);
    for (auto &b : out_bufs)
        freeShared(b);
    color = Tex();
    motion = Tex();
    depth = Tex();
    for (auto &t : outs)
        t = Tex();
    width = height = 0;
    sized = false;
}

// ---- neural rendering ----
// Two routes to DLSS 5 Neural Rendering. NVIDIA: NGX feature 18 from nvngx_dlssnr.dll,
// called through the gate (initGate). Elsewhere NeuralHub does not answer feature 18, so this
// creates a DLSS Super Resolution feature in DLAA mode (input size = output size), which
// NeuralHub serves with its upscaling provider and runs its neural rendering pass in front
// of, the same route as the DLSS 5 video players. Video has no depth and no motion vectors:
// the depth is a constant 0.5 and the motion vectors are zero.

bool Dx12Fg::Impl::nrCreateFeature()
{
    const unsigned int w = static_cast<unsigned int>(nr_cfg.width), h = static_cast<unsigned int>(nr_cfg.height);
    if (gate_ready) {
        // The model reads these at create time; feature 18 always runs at the input size.
        // Presets: 1 = CC (CC_Control_History_Blend weights), 0 / 2 / 3 = HT (hnet-almond-eagle).
        const unsigned int preset = nr_cfg.model == 1 ? 0u : 1u;
        nr_params->Reset();
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_Width, w);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_Height, h);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_OutWidth, w);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_OutHeight, h);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_MaxQuality);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_CreationNodeMask, 1);
        NVSDK_NGX_Parameter_SetUI(nr_params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Enabled", 1);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Width", w);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Height", h);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Hint.Render.Preset", preset);
        NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.Intensity", 1.0f);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Style", 0);
        NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.LocalStructureStrength", 1.0f);
        NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.LocalToneStrength", 1.0f);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.UseAutoMask", 0);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.UICorrection", 1);
        Slot *slot = nullptr;
        if (!beginList(&slot))
            return false;
        stage = "nvngx_dlssnr CreateFeature";
        const auto r = static_cast<NVSDK_NGX_Result>(
            gate.create(list.Get(), kFeatureNeuralRendering, nr_params, reinterpret_cast<void **>(&nr_handle)));
        const bool executed = endList(*slot) && waitIdle();
        if (NVSDK_NGX_FAILED(r) || !nr_handle) {
            log(1, "creating DLSS 5 neural rendering (feature 18) failed: %s (0x%08x) %s", ngxResultString(r),
                static_cast<unsigned>(r), gate.fault());
            nr_handle = nullptr;
            return false;
        }
        return executed && (!nrUpscales() || nrCreateUpscaler());
    }
    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = w;
    cp.Feature.InHeight = h;
    cp.Feature.InTargetWidth = static_cast<unsigned int>(nr_cfg.out_width);
    cp.Feature.InTargetHeight = static_cast<unsigned int>(nr_cfg.out_height);
    cp.Feature.InPerfQualityValue = qualityFor(nr_cfg.width, nr_cfg.out_width);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                              | (nr_cfg.hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0);
    Slot *slot = nullptr;
    if (!beginList(&slot))
        return false;
    stage = "NGX_D3D12_CREATE_DLSS_EXT";
    const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(list.Get(), 1, 1, &nr_handle, nr_params, &cp);
    const bool executed = endList(*slot) && waitIdle();
    if (NVSDK_NGX_FAILED(r) || !nr_handle) {
        log(1, "creating DLSS Super Resolution (D3D12, neural rendering) failed: %s (0x%08x)", ngxResultString(r),
            static_cast<unsigned>(r));
        nr_handle = nullptr;
        return false;
    }
    return executed;
}

// NVIDIA upscaling: NVIDIA's DLSS Super Resolution (nvngx_dlss.dll) from feature 18's output.
bool Dx12Fg::Impl::nrCreateUpscaler()
{
    if (!nr_sr_params && (NVSDK_NGX_FAILED(NVSDK_NGX_D3D12_AllocateParameters(&nr_sr_params)) || !nr_sr_params)) {
        log(1, "NGX parameter allocation for the upscaler failed");
        return false;
    }
    NVSDK_NGX_DLSS_Create_Params cp = {};
    cp.Feature.InWidth = static_cast<unsigned int>(nr_cfg.width);
    cp.Feature.InHeight = static_cast<unsigned int>(nr_cfg.height);
    cp.Feature.InTargetWidth = static_cast<unsigned int>(nr_cfg.out_width);
    cp.Feature.InTargetHeight = static_cast<unsigned int>(nr_cfg.out_height);
    cp.Feature.InPerfQualityValue = qualityFor(nr_cfg.width, nr_cfg.out_width);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                              | (nr_cfg.hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0);
    Slot *slot = nullptr;
    if (!beginList(&slot))
        return false;
    stage = "NGX_D3D12_CREATE_DLSS_EXT (upscaling after neural rendering)";
    const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(list.Get(), 1, 1, &nr_sr_handle, nr_sr_params, &cp);
    const bool executed = endList(*slot) && waitIdle();
    if (NVSDK_NGX_FAILED(r) || !nr_sr_handle) {
        log(1, "creating DLSS Super Resolution for upscaling failed: %s (0x%08x)", ngxResultString(r),
            static_cast<unsigned>(r));
        nr_sr_handle = nullptr;
        return false;
    }
    return executed;
}

bool Dx12Fg::Impl::createProxyPipeline()
{
    if (cs_encode && cs_compose && cs_heap)
        return true;
    if (cs_failed)
        return false;
    cs_failed = true; // until everything below succeeds
    stage = "HDR proxy shaders";
    HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    using SerializeFn = HRESULT(WINAPI *)(const D3D12_ROOT_SIGNATURE_DESC *, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob **,
                                          ID3DBlob **);
    const auto serialize = reinterpret_cast<SerializeFn>(GetProcAddress(d3d12_dll, "D3D12SerializeRootSignature"));
    if (!compile || !serialize) {
        log(1, "HDR neural rendering on NVIDIA needs d3dcompiler_47.dll and D3D12SerializeRootSignature");
        return false;
    }
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;
    D3D12_ROOT_PARAMETER rp[2] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[0].DescriptorTable.NumDescriptorRanges = 2;
    rp[0].DescriptorTable.pDescriptorRanges = ranges;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[1].Constants.ShaderRegister = 0;
    rp[1].Constants.Num32BitValues = 4;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2;
    rsd.pParameters = rp;
    ComPtr<ID3DBlob> blob, errors;
    if (FAILED(serialize(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors))
        || FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&cs_root)))) {
        log(1, "could not create the HDR proxy root signature");
        return false;
    }
    for (int i = 0; i < 2; i++) {
        const char *entry = i == 0 ? "encode" : "compose";
        ComPtr<ID3DBlob> code;
        errors.Reset();
        if (FAILED(compile(kProxyShaders, sizeof(kProxyShaders) - 1, "dx12fg_proxy", nullptr, nullptr, entry, "cs_5_0",
                           D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
            log(1, "compiling the HDR proxy shader %s failed: %s", entry,
                errors ? static_cast<const char *>(errors->GetBufferPointer()) : "no message");
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = cs_root.Get();
        pd.CS.pShaderBytecode = code->GetBufferPointer();
        pd.CS.BytecodeLength = code->GetBufferSize();
        ComPtr<ID3D12PipelineState> &target = i == 0 ? cs_encode : cs_compose;
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&target)))) {
            log(1, "could not create the HDR proxy pipeline %s", entry);
            return false;
        }
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 8;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cs_heap)))) {
        log(1, "could not create the HDR proxy descriptor heap");
        return false;
    }
    cs_increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    cs_failed = false;
    return true;
}

// Descriptors 0-3: encode (t0 = linear frame, u0 = proxy; t1 / t2 unused), 4-7: compose
// (t0 = linear frame, t1 = proxy, t2 = the model's proxy, u0 = composed frame).
void Dx12Fg::Impl::writeProxyDescriptors()
{
    const D3D12_CPU_DESCRIPTOR_HANDLE base = cs_heap->GetCPUDescriptorHandleForHeapStart();
    auto at = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = base;
        h.ptr += static_cast<SIZE_T>(i) * cs_increment;
        return h;
    };
    auto srv = [&](Tex &t, UINT i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
        d.Format = t.format;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(t.res.Get(), &d, at(i));
    };
    auto uav = [&](Tex &t, UINT i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
        d.Format = t.format;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(t.res.Get(), nullptr, &d, at(i));
    };
    srv(nr_in, 0);
    srv(nr_in, 1);
    srv(nr_in, 2);
    uav(nr_proxy_in, 3);
    srv(nr_in, 4);
    srv(nr_proxy_in, 5);
    srv(nr_proxy_out, 6);
    uav(nrUpscales() ? nr_mid : nr_out, 7);
}

void Dx12Fg::Impl::dispatchProxy(bool compose)
{
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier(nr_in, srv);
    if (compose) {
        barrier(nr_proxy_in, srv);
        barrier(nr_proxy_out, srv);
        barrier(nrUpscales() ? nr_mid : nr_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    } else {
        barrier(nr_proxy_in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12DescriptorHeap *heaps[] = { cs_heap.Get() };
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(cs_root.Get());
    list->SetPipelineState(compose ? cs_compose.Get() : cs_encode.Get());
    D3D12_GPU_DESCRIPTOR_HANDLE table = cs_heap->GetGPUDescriptorHandleForHeapStart();
    table.ptr += static_cast<UINT64>(compose ? 4 : 0) * cs_increment;
    list->SetComputeRootDescriptorTable(0, table);
    const UINT w = static_cast<UINT>(nr_cfg.width), h = static_cast<UINT>(nr_cfg.height);
    const float max_out = 10000.0f / 203.0f; // PQ's ceiling, in units of SDR white
    UINT constants[4] = { w, h, 0, 0 };
    std::memcpy(&constants[2], &max_out, sizeof(max_out));
    list->SetComputeRoot32BitConstants(1, 4, constants, 0);
    list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
}

bool Dx12Fg::Impl::nrEvaluate(bool reset, int log_level)
{
    const auto srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    const unsigned int w = static_cast<unsigned int>(nr_cfg.width), h = static_cast<unsigned int>(nr_cfg.height);
    barrier(nr_depth, srv);
    barrier(nr_motion, srv);
    if (gate_ready) {
        // feature 18 at the input size; HDR through the proxy; then the upscaler if any.
        const bool proxy = nrUsesProxy();
        if (proxy)
            dispatchProxy(false);
        Tex &model_in = proxy ? nr_proxy_in : nr_in;
        Tex &model_out = proxy ? nr_proxy_out : nrUpscales() ? nr_mid : nr_out;
        barrier(model_in, srv);
        barrier(model_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Color", model_in.res.Get());
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Depth", nr_depth.res.Get());
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.MVec", nr_motion.res.Get());
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Output", model_out.res.Get());
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Enabled", 1);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Reset", reset ? 1 : 0);
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.DepthInverted", 0);
        for (const char *name : { "Color", "Output", "Depth", "MVec" }) {
            const std::string base = std::string("DLSSNR.") + name + "Subrect";
            NVSDK_NGX_Parameter_SetUI(nr_params, (base + "BaseX").c_str(), 0);
            NVSDK_NGX_Parameter_SetUI(nr_params, (base + "BaseY").c_str(), 0);
            NVSDK_NGX_Parameter_SetUI(nr_params, (base + "Width").c_str(), w);
            NVSDK_NGX_Parameter_SetUI(nr_params, (base + "Height").c_str(), h);
        }
        NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.MVecScaleX", 1.0f);
        NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.MVecScaleY", 1.0f);
        stage = "nvngx_dlssnr EvaluateFeature";
        auto r = static_cast<NVSDK_NGX_Result>(gate.evaluate(list.Get(), nr_handle, nr_params));
        if (NVSDK_NGX_FAILED(r)) {
            log(log_level, "evaluating DLSS 5 neural rendering (feature 18) failed: %s (0x%08x) %s", ngxResultString(r),
                static_cast<unsigned>(r), gate.fault());
            return false;
        }
        if (proxy)
            dispatchProxy(true);
        if (!nrUpscales())
            return true;
        barrier(nr_mid, srv);
        barrier(nr_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        NVSDK_NGX_D3D12_DLSS_Eval_Params up = {};
        up.Feature.pInColor = nr_mid.res.Get();
        up.Feature.pInOutput = nr_out.res.Get();
        up.pInDepth = nr_depth.res.Get();
        up.pInMotionVectors = nr_motion.res.Get();
        up.InRenderSubrectDimensions = { w, h };
        up.InReset = reset ? 1 : 0;
        up.InMVScaleX = 1.0f;
        up.InMVScaleY = 1.0f;
        up.InPreExposure = 1.0f;
        up.InExposureScale = 1.0f;
        up.InFrameTimeDeltaInMsec = 16.7f;
        stage = "NGX_D3D12_EVALUATE_DLSS_EXT (upscaling after neural rendering)";
        r = NGX_D3D12_EVALUATE_DLSS_EXT(list.Get(), nr_sr_handle, nr_sr_params, &up);
        if (NVSDK_NGX_FAILED(r)) {
            log(log_level, "evaluating DLSS Super Resolution (upscaling) failed: %s (0x%08x)", ngxResultString(r),
                static_cast<unsigned>(r));
            return false;
        }
        return true;
    }
    barrier(nr_in, srv);
    barrier(nr_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    NVSDK_NGX_D3D12_DLSS_Eval_Params ep = {};
    ep.Feature.pInColor = nr_in.res.Get();
    ep.Feature.pInOutput = nr_out.res.Get();
    ep.pInDepth = nr_depth.res.Get();
    ep.pInMotionVectors = nr_motion.res.Get();
    ep.InJitterOffsetX = 0.0f;
    ep.InJitterOffsetY = 0.0f;
    ep.InRenderSubrectDimensions = { w, h };
    ep.InReset = reset ? 1 : 0;
    ep.InMVScaleX = 1.0f;
    ep.InMVScaleY = 1.0f;
    ep.InPreExposure = 1.0f;
    ep.InExposureScale = 1.0f;
    ep.InFrameTimeDeltaInMsec = 16.7f;
    stage = "NGX_D3D12_EVALUATE_DLSS_EXT";
    const NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSS_EXT(list.Get(), nr_handle, nr_params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        log(log_level, "evaluating DLSS Super Resolution (D3D12, neural rendering) failed: %s (0x%08x)",
            ngxResultString(r), static_cast<unsigned>(r));
        return false;
    }
    return true;
}

// The first evaluations build the model's pipelines (1-5 s with NeuralHub): done here, on
// a black frame, instead of during the stream.
bool Dx12Fg::Impl::nrWarmUp()
{
    std::vector<Tex *> textures = nrTextures();
    std::vector<Tex *> zero;
    for (Tex *t : textures) {
        if (t != &nr_depth)
            zero.push_back(t);
    }
    if (!fill(zero, &nr_depth))
        return false;
    for (int i = 0; i < 2; i++) {
        constexpr int kAttempts = 4;
        bool done = false;
        for (int attempt = 0; attempt < kAttempts && !done; attempt++) {
            if (attempt > 0) {
                log(0, "the NGX runtime is not open yet, presenting a new swapchain (attempt %d of %d)", attempt + 1,
                    kAttempts);
                Sleep(100u * static_cast<unsigned>(attempt));
                if (!createSwapchain())
                    return false;
            }
            std::vector<D3D12_RESOURCE_STATES> saved;
            for (Tex *t : textures)
                saved.push_back(t->state);
            Slot *slot = nullptr;
            if (!beginList(&slot))
                return false;
            if (!nrEvaluate(true, attempt + 1 < kAttempts ? 0 : 1)) {
                list->Close();
                for (size_t k = 0; k < textures.size(); k++)
                    textures[k]->state = saved[k];
                continue;
            }
            if (!endList(*slot) || !waitIdle())
                return false;
            done = true;
        }
        if (!done)
            return false;
    }
    return true;
}

void Dx12Fg::Impl::destroyNr()
{
    if (queue)
        waitIdle();
    if (nr_handle) {
        if (gate_ready)
            gate.release(nr_handle);
        else
            NVSDK_NGX_D3D12_ReleaseFeature(nr_handle);
    }
    nr_handle = nullptr;
    if (nr_sr_handle)
        NVSDK_NGX_D3D12_ReleaseFeature(nr_sr_handle);
    nr_sr_handle = nullptr;
    freeShared(nr_in_buf);
    freeShared(nr_out_buf);
    nr_in = Tex();
    nr_out = Tex();
    nr_depth = Tex();
    nr_motion = Tex();
    nr_mid = Tex();
    nr_proxy_in = Tex();
    nr_proxy_out = Tex();
    nr_cfg = NrConfig();
    nr_sized = false;
    nr_query_done = 0;
}

Dx12Fg::Dx12Fg(const VulkanDevice &vk, LogFn log, void *user)
    : impl(std::make_unique<Impl>())
{
    impl->vk = vk;
    impl->log_fn = log;
    impl->log_user = user;
}

Dx12Fg::~Dx12Fg()
{
    Impl &d = *impl;
    d.destroySized();
    d.destroyNr();
    if (d.semaphore)
        d.vkDestroySemaphore(d.vk.device, d.semaphore, nullptr);
    if (d.params)
        NVSDK_NGX_D3D12_DestroyParameters(d.params);
    if (d.gate_ready && d.device)
        d.gate.shutdown(d.device.Get());
    if (d.nr_params)
        NVSDK_NGX_D3D12_DestroyParameters(d.nr_params);
    if (d.nr_sr_params)
        NVSDK_NGX_D3D12_DestroyParameters(d.nr_sr_params);
    if (d.ngx_up && d.device)
        NVSDK_NGX_D3D12_Shutdown1(d.device.Get());
    d.swapchain.Reset();
    if (d.hwnd)
        DestroyWindow(d.hwnd);
    d.shared_fence.Reset();
    d.list.Reset();
    for (auto &slot : d.slots)
        slot.allocator.Reset();
    d.queue.Reset();
    d.idle_fence.Reset();
    if (d.idle_event)
        CloseHandle(d.idle_event);
    d.device.Reset();
    d.factory.Reset();
    // dxgi.dll / d3d12.dll stay loaded: other parts of the process use them too.
}

bool Dx12Fg::init()
{
    Impl &d = *impl;
    if (!d.loadVulkan() || !d.createDevice())
        return false;
    // Give the NGX provider time to hook the swapchain calls after the first device.
    Sleep(60);
    return d.createSwapchain() && d.initNgx() && d.createSharedFence();
}

int Dx12Fg::maxGenerated() const
{
    return impl->max_generated;
}

bool Dx12Fg::sizeMatches(int w, int h) const
{
    return impl->sized && impl->width == w && impl->height == h;
}

void Dx12Fg::invalidate()
{
    impl->destroySized();
}

bool Dx12Fg::ensureSize(int w, int h)
{
    Impl &d = *impl;
    if (sizeMatches(w, h))
        return true;
    d.destroySized();
    if (!d.fg_available) {
        d.log(1, "DLSS Frame Generation is not available through NGX D3D12 on this GPU");
        return false;
    }
    d.width = w;
    d.height = h;
    d.stage = "textures";
    bool ok = d.makeTex(d.color, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 4) &&
              d.makeTex(d.motion, w, h, DXGI_FORMAT_R16G16_FLOAT, 4) && d.makeTex(d.depth, w, h, DXGI_FORMAT_R32_FLOAT, 4);
    for (int i = 0; ok && i < d.max_generated; i++)
        ok = d.makeTex(d.outs[static_cast<size_t>(i)], w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    d.stage = "shared buffers";
    ok = ok && d.makeShared(d.color_buf, d.color.total) && d.makeShared(d.motion_buf, d.motion.total);
    for (int i = 0; ok && i < d.max_generated; i++)
        ok = d.makeShared(d.out_bufs[static_cast<size_t>(i)], d.outs[static_cast<size_t>(i)].total);
    if (!ok) {
        d.destroySized();
        return false;
    }
    if (!d.createFeature()) {
        d.destroySized();
        return false;
    }
    d.log(0, "warming up the D3D12 runtime (two evaluations, about 0.4 s)");
    if (!d.warmUp()) {
        d.log(1, "the warm-up evaluations failed");
        d.destroySized();
        return false;
    }
    d.sized = true;
    d.log(0, "DLSS Frame Generation (D3D12) ready: %dx%d", w, h);
    return true;
}

VkSemaphore Dx12Fg::semaphore() const
{
    return impl->semaphore;
}

uint64_t Dx12Fg::lastValue() const
{
    return impl->last_value;
}

void Dx12Fg::setLastValue(uint64_t value)
{
    impl->last_value = value;
}

void Dx12Fg::recordColor(VkCommandBuffer cmd, VkImage image)
{
    Impl &d = *impl;
    VkBufferImageCopy region = {};
    region.bufferRowLength = d.color.footprint.Footprint.RowPitch / d.color.bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.width), static_cast<uint32_t>(d.height), 1 };
    d.vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, d.color_buf.buffer, 1, &region);
}

void Dx12Fg::recordMotion(VkCommandBuffer cmd, VkImage image)
{
    Impl &d = *impl;
    VkBufferImageCopy region = {};
    region.bufferRowLength = d.motion.footprint.Footprint.RowPitch / d.motion.bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.width), static_cast<uint32_t>(d.height), 1 };
    d.vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, d.motion_buf.buffer, 1, &region);
}

void Dx12Fg::recordDownload(VkCommandBuffer cmd, VkImage image, int index)
{
    Impl &d = *impl;
    const Impl::Tex &out = d.outs[static_cast<size_t>(index)];
    VkBufferImageCopy region = {};
    region.bufferRowLength = out.footprint.Footprint.RowPitch / out.bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.width), static_cast<uint32_t>(d.height), 1 };
    d.vkCmdCopyBufferToImage(cmd, d.out_bufs[static_cast<size_t>(index)].buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

bool Dx12Fg::generate(int count, uint64_t wait_value, uint64_t signal_value, bool reset, bool use_motion,
                      float mvec_scale_x, float mvec_scale_y)
{
    Impl &d = *impl;
    count = std::clamp(count, 1, d.max_generated);
    d.stage = "queue wait";
    // GPU side wait for the Vulkan copies; the fence value is set by Vulkan.
    bool ok = d.sized && d.handle && SUCCEEDED(d.queue->Wait(d.shared_fence.Get(), wait_value));
    if (ok) {
        // Texture states as the commands will leave them; restored if recording fails.
        const auto saved_color = d.color.state;
        const auto saved_motion = d.motion.state;
        const auto saved_depth = d.depth.state;
        std::array<D3D12_RESOURCE_STATES, kMaxOutputs> saved_outs = {};
        for (size_t i = 0; i < saved_outs.size(); i++)
            saved_outs[i] = d.outs[i].state;
        Impl::Slot *slot = nullptr;
        const bool begun = d.beginList(&slot);
        ok = begun;
        if (ok) {
            d.copyBufferToTex(d.color, d.color_buf.res.Get());
            if (use_motion)
                d.copyBufferToTex(d.motion, d.motion_buf.res.Get());
            ok = d.evaluate(count, reset, mvec_scale_x, mvec_scale_y);
            for (int i = 0; ok && i < count; i++)
                d.copyTexToBuffer(d.out_bufs[static_cast<size_t>(i)].res.Get(), d.outs[static_cast<size_t>(i)]);
        }
        if (ok) {
            d.stage = "ExecuteCommandLists";
            ok = d.endList(*slot);
        } else if (begun) {
            // Recording failed: nothing of this list runs.
            d.list->Close();
        }
        if (!ok) {
            d.color.state = saved_color;
            d.motion.state = saved_motion;
            d.depth.state = saved_depth;
            for (size_t i = 0; i < saved_outs.size(); i++)
                d.outs[i].state = saved_outs[i];
        }
    }
    return d.signalShared(signal_value) && ok;
}

// Always signalled, so the Vulkan work waiting for it cannot hang.
bool Dx12Fg::Impl::signalShared(uint64_t value)
{
    stage = "queue signal";
    if (SUCCEEDED(queue->Signal(shared_fence.Get(), value)))
        return true;
    // The device is probably lost; let the Vulkan side go on from the host.
    VkSemaphoreSignalInfo signal_info = {};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = semaphore;
    signal_info.value = value;
    vkSignalSemaphore(vk.device, &signal_info);
    log(1, "the D3D12 queue could not signal the shared fence");
    return false;
}

bool Dx12Fg::fgAvailable() const
{
    return impl->fg_available;
}

bool Dx12Fg::nrAvailable() const
{
    // On NVIDIA the DLSS Super Resolution route would only be DLAA, not neural rendering.
    return impl->gate_ready || (impl->adapter_vendor != kVendorNvidia && impl->sr_available);
}

bool Dx12Fg::nrCanUpscale() const
{
    return impl->gate_ready ? impl->sr_available : nrAvailable();
}

bool Dx12Fg::nrMatches(const NrConfig &config) const
{
    return impl->nr_sized && impl->nr_cfg == config;
}

void Dx12Fg::nrInvalidate()
{
    impl->destroyNr();
}

bool Dx12Fg::nrEnsure(const NrConfig &config)
{
    Impl &d = *impl;
    if (nrMatches(config))
        return true;
    d.destroyNr();
    if (!nrAvailable()) {
        d.log(1, d.adapter_vendor == kVendorNvidia
                     ? "DLSS 5 neural rendering is not set up (nvngx_dlssnr.dll next to chiaki.exe, see above)"
                     : "DLSS Super Resolution is not available through NGX D3D12 on this GPU, no neural rendering");
        return false;
    }
    if (config.width <= 0 || config.height <= 0 || config.out_width < config.width
        || config.out_height < config.height) {
        d.log(1, "neural rendering: invalid size %dx%d -> %dx%d", config.width, config.height, config.out_width,
              config.out_height);
        return false;
    }
    d.nr_cfg = config;
    const bool upscale = d.nrUpscales();
    if (upscale && !nrCanUpscale()) {
        d.log(1, "neural rendering: no upscaler after DLSS 5 (nvngx_dlss.dll next to chiaki.exe), keeping %dx%d",
              config.width, config.height);
        d.nr_cfg = NrConfig();
        return false;
    }
    const int w = config.width, h = config.height, ow = config.out_width, oh = config.out_height;
    const DXGI_FORMAT color = config.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    const UINT bpt = config.hdr ? 8 : 4;
    d.stage = "neural rendering textures";
    bool ok = d.makeTex(d.nr_in, w, h, color, bpt) && d.makeTex(d.nr_out, ow, oh, color, bpt)
              && d.makeTex(d.nr_depth, w, h, DXGI_FORMAT_R32_FLOAT, 4)
              && d.makeTex(d.nr_motion, w, h, DXGI_FORMAT_R16G16_FLOAT, 4);
    if (ok && d.gate_ready && upscale)
        ok = d.makeTex(d.nr_mid, w, h, color, bpt);
    if (ok && d.nrUsesProxy()) {
        ok = d.makeTex(d.nr_proxy_in, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 4)
             && d.makeTex(d.nr_proxy_out, w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 4) && d.createProxyPipeline();
        if (ok)
            d.writeProxyDescriptors();
    }
    d.stage = "neural rendering shared buffers";
    ok = ok && d.makeShared(d.nr_in_buf, d.nr_in.total) && d.makeShared(d.nr_out_buf, d.nr_out.total);
    if (!ok || !d.nrCreateFeature()) {
        d.destroyNr();
        return false;
    }
    d.log(0, "warming up neural rendering (the first evaluations build the model, a few seconds)");
    if (!d.nrWarmUp()) {
        d.log(1, "the neural rendering warm-up evaluations failed");
        d.destroyNr();
        return false;
    }
    // Timing is optional.
    if (!d.nr_queries && SUCCEEDED(d.queue->GetTimestampFrequency(&d.timestamp_frequency))) {
        D3D12_QUERY_HEAP_DESC qd = {};
        qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qd.Count = 2;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 2 * sizeof(UINT64);
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&d.nr_queries))) ||
            FAILED(d.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&d.nr_query_readback)))) {
            d.nr_queries.Reset();
            d.nr_query_readback.Reset();
        }
    }
    d.nr_query_done = 0;
    d.nr_gpu_ms = 0.0;
    d.nr_sized = true;
    std::string route;
    if (d.gate_ready) {
        route = std::string("NGX feature 18 ") + (config.model == 1 ? "HT" : "CC");
        if (d.nrUsesProxy())
            route += " + HDR proxy";
        if (upscale)
            route += std::string(" + DLSS Super Resolution ") + qualityName(qualityFor(w, ow));
    } else {
        route = std::string("DLSS Super Resolution ") + qualityName(qualityFor(w, ow))
                + " through NeuralHub (DLSS 5 in front of the upscaler)";
    }
    d.log(0, "neural rendering ready: %dx%d -> %dx%d, %s, %s", w, h, ow, oh, config.hdr ? "HDR" : "SDR",
          route.c_str());
    return true;
}

float Dx12Fg::nrGpuTimeMs() const
{
    return static_cast<float>(impl->nr_gpu_ms);
}

void Dx12Fg::recordNrInput(VkCommandBuffer cmd, VkImage image)
{
    Impl &d = *impl;
    VkBufferImageCopy region = {};
    region.bufferRowLength = d.nr_in.footprint.Footprint.RowPitch / d.nr_in.bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.nr_cfg.width), static_cast<uint32_t>(d.nr_cfg.height), 1 };
    d.vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, d.nr_in_buf.buffer, 1, &region);
}

void Dx12Fg::recordNrOutput(VkCommandBuffer cmd, VkImage image)
{
    Impl &d = *impl;
    VkBufferImageCopy region = {};
    region.bufferRowLength = d.nr_out.footprint.Footprint.RowPitch / d.nr_out.bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.nr_cfg.out_width), static_cast<uint32_t>(d.nr_cfg.out_height), 1 };
    d.vkCmdCopyBufferToImage(cmd, d.nr_out_buf.buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

bool Dx12Fg::neuralRender(uint64_t wait_value, uint64_t signal_value, bool reset)
{
    Impl &d = *impl;
    d.stage = "queue wait";
    bool ok = d.nr_sized && d.nr_handle && SUCCEEDED(d.queue->Wait(d.shared_fence.Get(), wait_value));
    if (ok) {
        const std::vector<Impl::Tex *> textures = d.nrTextures();
        std::vector<D3D12_RESOURCE_STATES> saved;
        for (Impl::Tex *t : textures)
            saved.push_back(t->state);
        // Timestamps of an earlier evaluation are read once its list has finished; new
        // ones are only written when the readback buffer is free.
        bool timed = false;
        if (d.nr_queries) {
            const uint64_t completed = d.idle_fence->GetCompletedValue();
            if (d.nr_query_done && completed >= d.nr_query_done) {
                void *p = nullptr;
                const D3D12_RANGE range = { 0, 2 * sizeof(UINT64) };
                if (SUCCEEDED(d.nr_query_readback->Map(0, &range, &p))) {
                    const UINT64 *ts = static_cast<const UINT64 *>(p);
                    if (ts[1] > ts[0] && d.timestamp_frequency) {
                        const double ms = static_cast<double>(ts[1] - ts[0]) * 1000.0 /
                                          static_cast<double>(d.timestamp_frequency);
                        d.nr_gpu_ms = d.nr_gpu_ms > 0.0 ? d.nr_gpu_ms * 0.9 + ms * 0.1 : ms;
                    }
                    const D3D12_RANGE none = { 0, 0 };
                    d.nr_query_readback->Unmap(0, &none);
                }
                d.nr_query_done = 0;
            }
            timed = d.nr_query_done == 0;
        }
        Impl::Slot *slot = nullptr;
        const bool begun = d.beginList(&slot);
        ok = begun;
        if (ok) {
            d.copyBufferToTex(d.nr_in, d.nr_in_buf.res.Get());
            if (timed)
                d.list->EndQuery(d.nr_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            ok = d.nrEvaluate(reset);
            if (ok && timed) {
                d.list->EndQuery(d.nr_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                d.list->ResolveQueryData(d.nr_queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                                         d.nr_query_readback.Get(), 0);
            }
            if (ok)
                d.copyTexToBuffer(d.nr_out_buf.res.Get(), d.nr_out);
        }
        if (ok) {
            d.stage = "ExecuteCommandLists";
            ok = d.endList(*slot);
            if (ok && timed)
                d.nr_query_done = slot->done_value;
        } else if (begun) {
            d.list->Close();
        }
        if (!ok) {
            for (size_t k = 0; k < textures.size(); k++)
                textures[k]->state = saved[k];
        }
    }
    return d.signalShared(signal_value) && ok;
}

void Dx12Fg::markCrashed()
{
    impl->crashed = true;
}

bool Dx12Fg::crashed() const
{
    return impl->crashed;
}

std::shared_ptr<Dx12Fg> Dx12Fg::shared(const VulkanDevice &vk, LogFn log, void *user)
{
    static std::mutex mutex;
    static std::weak_ptr<Dx12Fg> instance;
    static VkDevice instance_device = VK_NULL_HANDLE;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto existing = instance.lock()) {
        if (instance_device == vk.device)
            return existing->crashed() ? nullptr : existing;
    }
    auto created = std::make_shared<Dx12Fg>(vk, log, user);
    if (!created->init())
        return nullptr;
    instance = created;
    instance_device = vk.device;
    return created;
}

const char *Dx12Fg::stage() const
{
    return impl->stage;
}
