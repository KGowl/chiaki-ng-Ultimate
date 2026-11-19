#include "dx12fg.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <vulkan/vulkan_win32.h>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers_dlssg_d3d.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

constexpr size_t kSlots = 4;
constexpr UINT kHiddenWindowSize = 1280;
// A project id of our own, as NGX wants one per application.
constexpr const char *kProjectId = "9d1c7a52-6a4e-4c36-8d4f-1f0a5f3b7c11";

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

    bool loadVulkan();
    bool createDevice();
    bool createSwapchain();
    bool initNgx();
    bool createSharedFence();
    bool makeTex(Tex &t, DXGI_FORMAT format, UINT bytes_per_texel);
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
    unsigned int available = 0;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &available))) {
        int available_i = 0;
        NVSDK_NGX_Parameter_GetI(caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &available_i);
        available = available_i > 0 ? 1 : 0;
    }
    if (!available) {
        log(1, "DLSS Frame Generation is not available through NGX D3D12 on this GPU");
        NVSDK_NGX_D3D12_DestroyParameters(caps);
        return false;
    }
    unsigned int multi_frame_max = 0;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetUI(caps, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &multi_frame_max))
        && multi_frame_max > 0)
        max_generated = std::min(static_cast<int>(multi_frame_max), kMaxOutputs);
    NVSDK_NGX_D3D12_DestroyParameters(caps);
    stage = "NVSDK_NGX_D3D12_AllocateParameters";
    r = NVSDK_NGX_D3D12_AllocateParameters(&params);
    if (NVSDK_NGX_FAILED(r) || !params) {
        params = nullptr;
        return false;
    }
    log(0, "DLSS Frame Generation (D3D12) available, up to %d generated frames", max_generated);
    return true;
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

bool Dx12Fg::Impl::makeTex(Tex &t, DXGI_FORMAT format, UINT bytes_per_texel)
{
    t.format = format;
    t.bytes_per_texel = bytes_per_texel;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = static_cast<UINT64>(width);
    rd.Height = static_cast<UINT>(height);
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = format;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&t.res));
    if (FAILED(hr)) {
        log(1, "could not create a %dx%d D3D12 texture (0x%08x)", width, height, static_cast<unsigned>(hr));
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

// The runtime builds its pipelines in the first two evaluations (about 0.2 s of CPU
// each): do it here, on dummy data, instead of during the stream.
bool Dx12Fg::Impl::warmUp()
{
    // Dummy content: zeroed colour and motion, constant depth. An upload buffer big
    // enough for every texture's footprint, filled with zeros, and one for the depth.
    const UINT64 size = std::max({ color.total, motion.total, depth.total });
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
    ComPtr<ID3D12Resource> upload_depth;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&upload_zero))) ||
        FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                               IID_PPV_ARGS(&upload_depth))))
        return false;
    void *p = nullptr;
    if (FAILED(upload_zero->Map(0, nullptr, &p)))
        return false;
    std::memset(p, 0, static_cast<size_t>(size));
    upload_zero->Unmap(0, nullptr);
    if (FAILED(upload_depth->Map(0, nullptr, &p)))
        return false;
    for (int y = 0; y < height; y++) {
        float *row = reinterpret_cast<float *>(static_cast<uint8_t *>(p) + static_cast<size_t>(y) * depth.footprint.Footprint.RowPitch);
        for (int x = 0; x < width; x++)
            row[x] = 0.5f; // a constant mid-range value keeps disocclusion logic neutral
    }
    upload_depth->Unmap(0, nullptr);

    Slot *slot = nullptr;
    if (!beginList(&slot))
        return false;
    copyBufferToTex(color, upload_zero.Get());
    copyBufferToTex(motion, upload_zero.Get());
    copyBufferToTex(depth, upload_depth.Get());
    for (auto &out : outs) {
        if (out.res)
            copyBufferToTex(out, upload_zero.Get());
    }
    if (!endList(*slot) || !waitIdle())
        return false;

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
    if (d.semaphore)
        d.vkDestroySemaphore(d.vk.device, d.semaphore, nullptr);
    if (d.params)
        NVSDK_NGX_D3D12_DestroyParameters(d.params);
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
    d.width = w;
    d.height = h;
    d.stage = "textures";
    bool ok = d.makeTex(d.color, DXGI_FORMAT_R8G8B8A8_UNORM, 4) && d.makeTex(d.motion, DXGI_FORMAT_R16G16_FLOAT, 4) &&
              d.makeTex(d.depth, DXGI_FORMAT_R32_FLOAT, 4);
    for (int i = 0; ok && i < d.max_generated; i++)
        ok = d.makeTex(d.outs[static_cast<size_t>(i)], DXGI_FORMAT_R8G8B8A8_UNORM, 4);
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
    // Always signalled, so the Vulkan work waiting for it cannot hang.
    d.stage = "queue signal";
    const bool signalled = SUCCEEDED(d.queue->Signal(d.shared_fence.Get(), signal_value));
    if (!signalled) {
        // The device is probably lost; let the Vulkan side go on from the host.
        VkSemaphoreSignalInfo signal_info = {};
        signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signal_info.semaphore = d.semaphore;
        signal_info.value = signal_value;
        d.vkSignalSemaphore(d.vk.device, &signal_info);
        d.log(1, "the D3D12 queue could not signal the shared fence");
    }
    return ok && signalled;
}

const char *Dx12Fg::stage() const
{
    return impl->stage;
}
