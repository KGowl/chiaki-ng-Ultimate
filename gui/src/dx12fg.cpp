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
#if defined(CHIAKI_HAVE_STREAMLINE)
#include <dxgi1_5.h>
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

#if defined(CHIAKI_HAVE_STREAMLINE)
// Streamline's API, forwarded to the sl.interposer.dll next to chiaki.exe, loaded at run time.
// sl.interposer.lib is not linked: it also exports D3D12CreateDevice / CreateDXGIFactory*, and
// linking it would hand out Streamline proxies where native interfaces are expected.
static HMODULE g_sl_dll = nullptr;
#define DX12FG_SL_FORWARD(name, params, args)                                                       \
    extern "C" sl::Result name params                                                               \
    {                                                                                               \
        static PFun_##name *fn = nullptr;                                                           \
        if (!fn && g_sl_dll)                                                                        \
            fn = reinterpret_cast<PFun_##name *>(GetProcAddress(g_sl_dll, #name));                  \
        return fn ? fn args : sl::Result::eErrorInitNotCalled;                                      \
    }
DX12FG_SL_FORWARD(slInit, (const sl::Preferences &pref, uint64_t sdkVersion), (pref, sdkVersion))
DX12FG_SL_FORWARD(slIsFeatureSupported, (sl::Feature feature, const sl::AdapterInfo &adapterInfo), (feature, adapterInfo))
DX12FG_SL_FORWARD(slSetTagForFrame,
                  (const sl::FrameToken &frame, const sl::ViewportHandle &viewport, const sl::ResourceTag *resources,
                   uint32_t numResources, sl::CommandBuffer *cmdBuffer),
                  (frame, viewport, resources, numResources, cmdBuffer))
DX12FG_SL_FORWARD(slSetConstants, (const sl::Constants &values, const sl::FrameToken &frame, const sl::ViewportHandle &viewport),
                  (values, frame, viewport))
DX12FG_SL_FORWARD(slGetFeatureRequirements, (sl::Feature feature, sl::FeatureRequirements &requirements),
                  (feature, requirements))
DX12FG_SL_FORWARD(slUpgradeInterface, (void **baseInterface), (baseInterface))
DX12FG_SL_FORWARD(slGetNativeInterface, (void *proxyInterface, void **baseInterface), (proxyInterface, baseInterface))
DX12FG_SL_FORWARD(slGetFeatureFunction, (sl::Feature feature, const char *functionName, void *&function),
                  (feature, functionName, function))
DX12FG_SL_FORWARD(slGetNewFrameToken, (sl::FrameToken * &token, const uint32_t *frameIndex), (token, frameIndex))
DX12FG_SL_FORWARD(slSetD3DDevice, (void *d3dDevice), (d3dDevice))
#undef DX12FG_SL_FORWARD
#endif

namespace {

// Instances with NGX's D3D12 side up (Dx12Fg::ngxUpInProcess). A crashed instance is never
// destroyed and stays counted, as its NGX state does.
std::atomic<int> g_ngx_d3d12_up{0};

constexpr size_t kSlots = 4;
constexpr UINT kHiddenWindowSize = 1280;
// A project id of our own, as NGX wants one per application.
constexpr const char *kProjectId = "9d1c7a52-6a4e-4c36-8d4f-1f0a5f3b7c11";
constexpr UINT kVendorNvidia = 0x10DE;
// DLSS Neural Rendering's NGX feature id (a reserved slot in the public headers).
constexpr int kFeatureNeuralRendering = 18;
// NGX's generic application id, for the snippet's own Init_Ext.
constexpr unsigned long long kNrApplicationId = 0x24480451ull;

double nowMs()
{
    static const LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return 1000.0 * static_cast<double>(c.QuadPart) / static_cast<double>(freq.QuadPart);
}

#if defined(CHIAKI_HAVE_STREAMLINE)
// A QueryPerformanceCounter value on nowMs()'s scale.
double qpcMs(uint64_t qpc)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return 1000.0 * static_cast<double>(qpc) / static_cast<double>(f.QuadPart);
}

// ---- DLSS Frame Generation through NVIDIA Streamline (Dx12Fg::Impl::sl_route) ----
// MFG Unlocker (github.com/dashdogy/MFG-Unlocker) opens DLSS Frame Generation on RTX 20 / RTX 30
// for Streamline only, not for NGX called directly. Streamline (manual hooking) owns a hidden
// window's swapchain: each real frame is copied into its back buffer and presented, and
// Streamline presents the generated frames on that swapchain itself, paced over the next frame
// interval. A hook on DXGI's Present copies every frame presented on that window back (real and
// generated alike) and the real ones are told apart by a marker row under the frame (alpha 0;
// DLSS-G writes alpha 255 into the frames it generates).
constexpr size_t kSlBackBuffers = 3;
constexpr size_t kSlRing = 2 * Dx12Fg::kMaxOutputs + 2; // frame copies: up to 2 frames' generated + real
constexpr size_t kSlCapLists = 8;
constexpr uint64_t kSlReadback = 64;      // marker pixels read back, one per presented frame
constexpr UINT64 kSlReadbackStride = 512; // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT
constexpr UINT64 kSlTimestampOffset = 256; // each copy's GPU timestamp, in its readback entry
constexpr uint32_t kSlIdMask = 0xffffff;  // frame id in the marker's RGB
enum SlSlotState { kSlFree, kSlCaptured, kSlTaken };

std::mutex g_sl_mutex; // start-up and hooks, once per process
bool g_sl_tried = false;
bool g_sl_started = false;
bool g_sl_hooked = false;
// The window Streamline generates frames on, and the queue Streamline presents it with (its
// own, seen when it creates its swapchain for that window).
std::atomic<HWND> g_sl_hwnd{nullptr};
std::atomic<ID3D12CommandQueue *> g_sl_present_queue{nullptr};
// The instance the frames presented on that window are copied for; the Present hook holds the
// mutex while it records a copy.
std::mutex g_sl_capture_mutex;
Dx12Fg::Impl *g_sl_capture_owner = nullptr;
// CHIAKI_DLSSG_SL_ALWAYS=1: what GetForegroundWindow returns inside Streamline (DLSS-G pauses
// while no window of the process has the focus).
HWND g_sl_focus_window = nullptr;
// CHIAKI_DLSSG_SL_TRACE=1: logs when each of Streamline's presents came and how long it took.
bool g_sl_trace = false;
using PresentFn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain *, UINT, UINT);
using Present1Fn = HRESULT(STDMETHODCALLTYPE *)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);
using CreateForHwndFn = HRESULT(STDMETHODCALLTYPE *)(IDXGIFactory2 *, IUnknown *, HWND, const DXGI_SWAP_CHAIN_DESC1 *,
                                                       const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *, IDXGIOutput *,
                                                       IDXGISwapChain1 **);
using CreateFn = HRESULT(STDMETHODCALLTYPE *)(IDXGIFactory *, IUnknown *, DXGI_SWAP_CHAIN_DESC *, IDXGISwapChain **);
PresentFn g_orig_present = nullptr;
Present1Fn g_orig_present1 = nullptr;
CreateForHwndFn g_orig_create_hwnd = nullptr;
CreateFn g_orig_create = nullptr;

std::wstring exeDir()
{
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir = exe;
    return dir.substr(0, dir.find_last_of(L'\\'));
}

bool fileExists(const std::wstring &path)
{
    return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;
}

// Whether DLSS Frame Generation goes through Streamline: when MFG Unlocker (its version.dll)
// is next to chiaki.exe; CHIAKI_DLSSG_ROUTE=sl / ngx decides otherwise. Needs sl.interposer.dll.
// Why DLSS Frame Generation goes through Streamline on this adapter, or nullptr for NGX directly.
// RTX 50 cards generate up to 4x through NGX themselves; through Streamline on the hidden window
// an RTX 5090 gave its frames only every 14-20 ms. `note` says why version.dll is not followed.
const char *slRouteReason(const char *adapter_name, const char **note)
{
    *note = nullptr;
    const std::wstring dir = exeDir();
    if (!fileExists(dir + L"\\sl.interposer.dll"))
        return nullptr;
    const char *route = std::getenv("CHIAKI_DLSSG_ROUTE");
    if (route && _stricmp(route, "sl") == 0)
        return "CHIAKI_DLSSG_ROUTE=sl";
    if (route && _stricmp(route, "ngx") == 0)
        return nullptr;
    if (!fileExists(dir + L"\\version.dll"))
        return nullptr;
    if (strstr(adapter_name, "RTX 50") || strstr(adapter_name, "Blackwell")) {
        *note = "MFG Unlocker's version.dll is next to chiaki.exe, but RTX 50 cards run DLSS Frame Generation "
                "through NGX directly (CHIAKI_DLSSG_ROUTE=sl for Streamline)";
        return nullptr;
    }
    return "MFG Unlocker's version.dll is next to chiaki.exe";
}
#endif

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
// NeuralHub's panel window on an HDR stream: the frame (1.0 = SDR white, 203 nits) in scRGB
// units (1.0 = 80 nits); max_out carries the scale.
[numthreads(8, 8, 1)]
void scrgb(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= size.x || id.y >= size.y)
        return;
    dst[id.xy] = float4(src0[id.xy].rgb * max_out, 1.0);
}
)";

// NeuralHub writes "overlay shown (...)" and "overlay hidden" to its log (dlssnr_on_amd.log
// next to the executable) when its in-game panel opens and closes; this follows the log from
// where it was when opened.
struct PanelLog
{
    HANDLE file = INVALID_HANDLE_VALUE;
    LONGLONG pos = 0;
    std::string carry;

    ~PanelLog()
    {
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
    }
    bool open(const std::wstring &path)
    {
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER size = {};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &size))
            return false;
        pos = size.QuadPart;
        return true;
    }
    // +1: the panel opened, -1: it closed (the last of the new lines decides), 0: neither.
    int poll()
    {
        LARGE_INTEGER size = {};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &size))
            return 0;
        if (size.QuadPart < pos) { // a new log
            pos = 0;
            carry.clear();
        }
        if (size.QuadPart == pos)
            return 0;
        const DWORD want = static_cast<DWORD>(std::min<LONGLONG>(size.QuadPart - pos, 1 << 18));
        std::string text = carry;
        text.resize(carry.size() + want);
        LARGE_INTEGER at = {};
        at.QuadPart = pos;
        DWORD got = 0;
        if (!SetFilePointerEx(file, at, nullptr, FILE_BEGIN) || !ReadFile(file, &text[carry.size()], want, &got, nullptr))
            return 0;
        text.resize(carry.size() + got);
        pos += got;
        const size_t nl = text.find_last_of('\n');
        if (nl == std::string::npos) {
            carry = text.size() < 4096 ? text : std::string();
            return 0;
        }
        carry = text.substr(nl + 1);
        text.resize(nl);
        const size_t shown = text.rfind("overlay shown"), hidden = text.rfind("overlay hidden");
        if (shown == std::string::npos && hidden == std::string::npos)
            return 0;
        if (hidden == std::string::npos)
            return 1;
        if (shown == std::string::npos)
            return -1;
        return shown > hidden ? 1 : -1;
    }
};

// Whether Windows shows HDR on this monitor ("Use HDR"). There scRGB 1.0 is 80 nits, so an HDR
// frame (1.0 = 203 nits) is scaled by 203/80; on an SDR desktop scRGB 1.0 is SDR white and the
// frame goes in unscaled (2.5 times too bright otherwise). Auto colour management also sets
// advancedColorEnabled on SDR monitors, with wideColorEnforced.
bool monitorShowsHdr(HMONITOR monitor)
{
    MONITORINFOEXW info = {};
    info.cbSize = sizeof(info);
    if (!monitor || !GetMonitorInfoW(monitor, &info))
        return false;
    UINT32 path_count = 0, mode_count = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) != ERROR_SUCCESS)
        return false;
    std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
    std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
    if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(), &mode_count, modes.data(), nullptr)
        != ERROR_SUCCESS)
        return false;
    for (UINT32 i = 0; i < path_count; i++) {
        DISPLAYCONFIG_SOURCE_DEVICE_NAME source = {};
        source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size = sizeof(source);
        source.header.adapterId = paths[i].sourceInfo.adapterId;
        source.header.id = paths[i].sourceInfo.id;
        if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS || wcscmp(source.viewGdiDeviceName, info.szDevice) != 0)
            continue;
        DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO color = {};
        color.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO;
        color.header.size = sizeof(color);
        color.header.adapterId = paths[i].targetInfo.adapterId;
        color.header.id = paths[i].targetInfo.id;
        if (DisplayConfigGetDeviceInfo(&color.header) != ERROR_SUCCESS)
            return false;
        return color.advancedColorEnabled && !color.wideColorEnforced;
    }
    return false;
}

LRESULT CALLBACK hiddenWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    // Shown as NeuralHub's panel window it never takes the focus from chiaki's window.
    if (msg == WM_MOUSEACTIVATE)
        return MA_NOACTIVATE;
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// NeuralHub (the NGX provider on AMD) hooks the window procedure of each swapchain window it sees
// presented on its device, one window at a time: it keeps a single saved procedure and never puts
// the previous window's back. Two windows presenting in turn leave it calling itself at the next
// message (a stack overflow in user32). chiaki's own window presents on the same device (AMD's
// Vulkan driver presents through a DXGI swapchain, and D3D12 has one device per adapter), so
// once the provider has hooked another window of this process, the hidden swapchain must never
// be presented again.
HMODULE moduleOf(LONG_PTR address)
{
    HMODULE module = nullptr;
    if (!address
        || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(address), &module))
        return nullptr;
    return module;
}

struct ProviderWindowScan
{
    HMODULE provider;
    HWND own;
    DWORD pid;
    HWND found;
};

bool providerHooked(HWND window, const ProviderWindowScan &scan)
{
    return window != scan.own && moduleOf(GetWindowLongPtrW(window, GWLP_WNDPROC)) == scan.provider;
}

BOOL CALLBACK providerWindowChild(HWND window, LPARAM lp)
{
    auto *scan = reinterpret_cast<ProviderWindowScan *>(lp);
    if (!providerHooked(window, *scan))
        return TRUE;
    scan->found = window;
    return FALSE;
}

BOOL CALLBACK providerWindowTop(HWND window, LPARAM lp)
{
    auto *scan = reinterpret_cast<ProviderWindowScan *>(lp);
    DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    if (pid != scan->pid)
        return TRUE;
    if (providerHooked(window, *scan))
        scan->found = window;
    else
        EnumChildWindows(window, providerWindowChild, lp);
    return scan->found ? FALSE : TRUE;
}

// A window of this process other than `own` whose procedure is NeuralHub's, or null.
HWND providerHookedWindow(HWND own)
{
    const HMODULE provider = GetModuleHandleW(L"neuralhub_host.dll");
    if (!provider)
        return nullptr;
    ProviderWindowScan scan = { provider, own, GetCurrentProcessId(), nullptr };
    EnumWindows(providerWindowTop, reinterpret_cast<LPARAM>(&scan));
    return scan.found;
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
    // Feature 18 passes after the first (nr_handle) and the textures between the passes.
    std::array<NVSDK_NGX_Handle *, kMaxNrPasses - 1> nr_more = {};
    std::array<Tex, 2> nr_chain;
    // Feature 18's live controls per pass: what the next evaluation sets, and what the last
    // one set (a difference starts a new temporal history in that pass and the ones after).
    std::array<NrTuning, kMaxNrPasses> nr_tuning, nr_tuning_set;
    bool nr_tuning_valid = false;
    ULONGLONG nr_tuning_logged = 0;
#if defined(CHIAKI_DX12FG_TEST_HOOKS)
    std::string nr_test_keys; // CHIAKI_DX12FG_NR_KEYS as last applied
    bool nrTestKeys();
#endif
    void nrSetKeys(int pass);
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

    // ---- NeuralHub's in-game panel (its key, End by default), NeuralHub route only ----
    // NeuralHub polls its keys and draws its panel when it sees its swapchain presented, so
    // the hidden swapchain is presented after every evaluation. Its log says when the panel
    // opens and closes; while it is open and chiaki's window has the focus, the window is
    // shown over the video and each frame is copied into the swapchain, under the panel.
    // NeuralHub keeps the panel's keys from the process's message loop while it is open.
    HWND panel_owner = nullptr;
    PanelLog panel_log;
    bool panel_log_tried = false;
    bool panel_open = false;   // NeuralHub's panel, as its log says
    bool panel_copy = false;   // this frame goes into the swapchain
    bool panel_shown = false;  // the window is on screen
    bool panel_styled = false; // the window has its popup styles
    bool panel_failed = false; // the swapchain could not take the frame's size: not shown
    bool panel_present_warned = false;
    RECT panel_rect = {};
    UINT panel_w = kHiddenWindowSize, panel_h = kHiddenWindowSize * 9 / 16;
    DXGI_FORMAT panel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
    Tex nr_panel; // HDR: the frame in scRGB units, copied into the swapchain
    bool panel_desc_ready = false;
    HMONITOR panel_monitor = nullptr; // where the panel is, checked for HDR every 2 s
    ULONGLONG panel_monitor_checked = 0;
    int panel_monitor_hdr = -1; // -1 not checked yet
    float panel_scrgb = 1.0f;   // HDR frames into scRGB: 203/80 on an HDR desktop, 1 on SDR
    ComPtr<ID3D12PipelineState> cs_scrgb;
    // Another window of the process presents on the provider's device (chiaki's own window,
    // see providerHookedWindow): the hidden swapchain is never presented, the provider's runtime
    // opens on that window's presents and NeuralHub draws its panel on it.
    bool shared_present = false;
    bool mayPresent();
    bool panelRoute() const { return !gate_ready && nr_sized && swapchain && !shared_present; }
    void panelUpdate();
    void panelRecordCopy();
    void panelPresent();
    void panelHide();
    bool panelResize(UINT w, UINT h, DXGI_FORMAT format);

    // ---- DLSS Frame Generation through Streamline (see kSlRing) ----
    bool sl_route = false;
#if defined(CHIAKI_HAVE_STREAMLINE)
    int sl_max_generated = 1;
    ID3D12Device *sl_device = nullptr;   // Streamline's proxy of `device`
    ComPtr<ID3D12CommandQueue> sl_queue; // proxy; `queue` is its native queue
    IDXGIFactory4 *sl_factory = nullptr; // Streamline's proxy of `factory`
    HWND sl_hwnd = nullptr;
    bool sl_tearing = false;
    ComPtr<IDXGISwapChain3> sl_swapchain; // proxy, width x (height + 1): the marker row at the bottom
    std::array<ComPtr<ID3D12Resource>, kSlBackBuffers> sl_buffers;
    int sl_count_set = 0;   // numFramesToGenerate Streamline has
    uint32_t sl_frame = 0;  // id of the last frame handed to Streamline (in its marker)
    ComPtr<ID3D12Resource> sl_marker; // upload: a marker row per command allocator slot
    uint8_t *sl_marker_ptr = nullptr;
    UINT64 sl_marker_stride = 0;
    // Copies of the frames presented on sl_hwnd, recorded by the Present hook under
    // g_sl_capture_mutex: a frame slot each (kSlTaken: until the shared timeline passes
    // free_after, when Vulkan has copied it out) and its marker pixel in sl_readback.
    struct SlSlot
    {
        Shared buf;
        int state = kSlFree;
        uint64_t free_after = 0;
    };
    std::array<SlSlot, kSlRing> sl_ring;
    ComPtr<ID3D12Resource> sl_readback;
    const uint8_t *sl_readback_ptr = nullptr;
    // GPU timestamps of the copies (when each frame was there) and the clocks' calibration.
    ComPtr<ID3D12QueryHeap> sl_ts_heap;
    UINT64 sl_ts_freq = 0, sl_ts_gpu0 = 0, sl_ts_cpu0 = 0;
    double sl_ts_calibrated = -1.0e9;
    ComPtr<ID3D12Fence> sl_cap_fence; // value = presents copied so far
    HANDLE sl_cap_event = nullptr;
    struct SlCapList
    {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        uint64_t value = 0;
    };
    std::array<SlCapList, kSlCapLists> sl_cap_lists;
    uint64_t sl_cap_seq = 0;
    size_t sl_slot_next = 0;
    struct SlRecord
    {
        uint64_t seq;
        int slot; // -1: no free slot, only the marker was read
    };
    std::deque<SlRecord> sl_records;
    bool sl_capture_on = false;
    bool sl_format_warned = false;
    // Sorting the copies (main thread, under g_sl_capture_mutex).
    int64_t sl_last_real = -1;    // id of the last real frame seen presented
    std::vector<int> sl_gens;     // generated frames presented after it
    std::vector<int> sl_ready;    // the waiting frame's generated frames, complete
    bool sl_ready_valid = false;
    bool sl_waiting = false;      // the last generate()'s frames are not taken yet
    int sl_count = 1;             // generated frames asked for it
    std::array<int, kMaxOutputs> sl_taken = {};
    double sl_t_generate = 0.0;   // when it was handed over
    double sl_interval = 16.7;    // between generate() calls, smoothed
    float sl_stream_interval = 0.0f; // the stream's frame interval (setFrameInterval), 0 unknown
    // When each copy was recorded (Streamline's present call, the GPU may still be generating the
    // frame then) and, for the generated frames being sorted and ready, also when their copy was
    // seen done (when they are there for chiaki).
    std::array<double, kSlReadback> sl_cap_time = {};
    struct SlTimes
    {
        double first_cap = 0.0, last_cap = 0.0, first_seen = 0.0, last_seen = 0.0;
    };
    SlTimes sl_gens_times, sl_ready_times;
    // Generated frames per real frame the GPU keeps up with. DLSS-G presents them one after the
    // other as each is done (on RTX 20 / RTX 30 about 8 ms each at 1080p), so all of them have to
    // be in before the next stream frame is due, or every frame after it comes later.
    int sl_cap = kMaxOutputs;
    double sl_first_ms = 0.0;   // generate() -> first generated frame presented, smoothed
    double sl_spacing_ms = 0.0; // between the generated frames of one real frame, smoothed
    uint32_t sl_late_bits = 0;  // the last frames, newest in bit 0: late or not
    int sl_room_frames = 0;     // frames in a row with room for one more generated frame
    int sl_settle = 0;          // frames not judged after a change
    int sl_probe_wait = 600;    // frames with room before one more is tried without evidence
    bool sl_probing = false;    // such a try is on
    int sl_probe_frames = 0;    // frames since it began
    // Statistics, logged every 10 s.
    uint64_t sl_stat_frames = 0, sl_stat_gens = 0, sl_stat_short = 0, sl_stat_timeouts = 0;
    uint64_t sl_stat_dropped = 0, sl_stat_skipped = 0;
    double sl_stat_wait_ms = 0.0;
    double sl_stat_logged = 0.0;
    int sl_traced = 0;
    // CHIAKI_DLSSG_SL_TRACE: per copy, when it was made (after the last generate()), how long the
    // present after it took, and the present's sync interval and flags.
    std::array<double, kSlReadback> sl_trace_t = {};
    std::array<float, kSlReadback> sl_trace_present = {};
    std::array<uint32_t, kSlReadback> sl_trace_mode = {};
    std::string sl_trace_line;
    int sl_trace_lines = 0;

    bool slStartRuntime();
    bool slAttachDevice();
    bool slSetup();
    bool slEnsureSize(int w, int h);
    void slDestroySized();
    void slResetSortingLocked();
    bool slSetOptions(int count);
    bool slPresentFrame(int count, bool reset, bool signal, uint64_t signal_value);
    bool slGenerate(int count, uint64_t wait_value, uint64_t signal_value, bool reset);
    uint64_t slCapture(IDXGISwapChain *sc, UINT sync, UINT flags);
    void slSortLocked();
    void slFreeLocked(int slot);
    double slFrameIntervalMs() const { return sl_stream_interval > 0.0f ? sl_stream_interval : sl_interval; }
    double slTimeoutMs() const { return std::clamp(slFrameIntervalMs() * 1.25 + 8.0, 20.0, 100.0); }
    void slAdaptLocked(int n, const SlTimes &times);
    bool slFramesReady();
    int slTakeFrames();
#endif

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
        for (Tex *t : { &nr_in, &nr_out, &nr_depth, &nr_motion, &nr_mid, &nr_proxy_in, &nr_proxy_out, &nr_chain[0],
                        &nr_chain[1], &nr_panel }) {
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
#if defined(CHIAKI_HAVE_STREAMLINE)
    // Streamline starts before the device exists (manual hooking: it hooks nothing itself).
    bool sl_wanted = false;
    if (adapter_vendor == kVendorNvidia) {
        const char *note = nullptr;
        if (const char *reason = slRouteReason(name, &note)) {
            log(0, "DLSS Frame Generation through Streamline (%s)", reason);
            sl_wanted = slStartRuntime();
        } else if (note) {
            log(0, "%s", note);
        }
    }
#endif
    HRESULT hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        log(1, "D3D12CreateDevice failed on %s (0x%08x)", name, static_cast<unsigned>(hr));
        return false;
    }
#if defined(CHIAKI_HAVE_STREAMLINE)
    // On the Streamline route the queue comes from Streamline's proxy device (its swapchain is
    // created with it); everything runs on that queue's native queue.
    if (sl_wanted && !slAttachDevice())
        log(1, "Streamline: DLSS Frame Generation goes through NGX instead");
#endif
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!queue && FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))))
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

#if defined(CHIAKI_HAVE_STREAMLINE)
namespace {

// Every Present of a DXGI swapchain comes through here (the hook is on the class's vtable): the
// frames presented on Streamline's window are copied for the instance that owns it.
// Returns 1 + the copy's sequence number with CHIAKI_DLSSG_SL_TRACE (0 otherwise).
uint64_t slCaptureHook(IDXGISwapChain *sc, UINT sync, UINT flags)
{
    if (flags & DXGI_PRESENT_TEST)
        return 0;
    const HWND hwnd = g_sl_hwnd.load();
    if (!hwnd)
        return 0;
    DXGI_SWAP_CHAIN_DESC desc = {};
    if (FAILED(sc->GetDesc(&desc)) || desc.OutputWindow != hwnd)
        return 0;
    std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
    if (!g_sl_capture_owner)
        return 0;
    return g_sl_capture_owner->slCapture(sc, sync, flags);
}

void slTracePresent(uint64_t traced, double ms)
{
    std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
    if (g_sl_capture_owner)
        g_sl_capture_owner->sl_trace_present[(traced - 1) % kSlReadback] = static_cast<float>(ms);
}

HRESULT STDMETHODCALLTYPE hookPresent(IDXGISwapChain *sc, UINT sync, UINT flags)
{
    const uint64_t traced = slCaptureHook(sc, sync, flags);
    const double t0 = traced ? nowMs() : 0.0;
    const HRESULT hr = g_orig_present(sc, sync, flags);
    if (traced)
        slTracePresent(traced, nowMs() - t0);
    return hr;
}

HRESULT STDMETHODCALLTYPE hookPresent1(IDXGISwapChain1 *sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS *params)
{
    const uint64_t traced = slCaptureHook(sc, sync, flags);
    const double t0 = traced ? nowMs() : 0.0;
    const HRESULT hr = g_orig_present1(sc, sync, flags, params);
    if (traced)
        slTracePresent(traced, nowMs() - t0);
    return hr;
}

// IDXGISwapChain::Present is slot 8 of the vtable, IDXGISwapChain1::Present1 slot 22.
bool swapchainHooked(IDXGISwapChain *sc)
{
    void **vt = *reinterpret_cast<void ***>(sc);
    return vt[8] == reinterpret_cast<void *>(&hookPresent) && vt[22] == reinterpret_cast<void *>(&hookPresent1);
}

// -1 not seen yet, 0 Streamline's swapchain is of another kind (its presents are not seen), 1 hooked
std::atomic<int> g_sl_swapchain_hooked{-1};

// Streamline creates its own swapchain for the frame generation window, on a queue of its own:
// the copies of what it presents go on that queue.
void noteSwapchain(IUnknown *device, HWND hwnd, IUnknown *created)
{
    if (!hwnd || !device || hwnd != g_sl_hwnd.load())
        return;
    ComPtr<ID3D12CommandQueue> q;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&q))) && g_sl_present_queue.load() != q.Get()) {
        q->AddRef(); // kept for the process: the Present hook may still use the one it replaces
        g_sl_present_queue.store(q.Get());
    }
    ComPtr<IDXGISwapChain> sc;
    if (created && SUCCEEDED(created->QueryInterface(IID_PPV_ARGS(&sc))))
        g_sl_swapchain_hooked.store(swapchainHooked(sc.Get()) ? 1 : 0);
}

HRESULT STDMETHODCALLTYPE hookCreateForHwnd(IDXGIFactory2 *factory, IUnknown *device, HWND hwnd,
                                            const DXGI_SWAP_CHAIN_DESC1 *desc, const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *fs,
                                            IDXGIOutput *output, IDXGISwapChain1 **sc)
{
    const HRESULT hr = g_orig_create_hwnd(factory, device, hwnd, desc, fs, output, sc);
    if (SUCCEEDED(hr) && sc)
        noteSwapchain(device, hwnd, *sc);
    return hr;
}

HRESULT STDMETHODCALLTYPE hookCreate(IDXGIFactory *factory, IUnknown *device, DXGI_SWAP_CHAIN_DESC *desc, IDXGISwapChain **sc)
{
    const HRESULT hr = g_orig_create(factory, device, desc, sc);
    if (SUCCEEDED(hr) && sc && desc)
        noteSwapchain(device, desc->OutputWindow, *sc);
    return hr;
}

bool patchSlot(void **vtable, int index, void *hook, void **orig)
{
    if (vtable[index] == hook)
        return true;
    DWORD old = 0;
    if (!VirtualProtect(&vtable[index], sizeof(void *), PAGE_EXECUTE_READWRITE, &old))
        return false;
    if (!*orig)
        *orig = vtable[index];
    vtable[index] = hook;
    VirtualProtect(&vtable[index], sizeof(void *), old, &old);
    return true;
}

HWND WINAPI hookGetForegroundWindow()
{
    return g_sl_focus_window;
}

// Points `func` imported from `dll` by `module` at `hook`; returns how many imports changed.
int patchImport(HMODULE module, const char *dll, const char *func, void *hook)
{
    if (!module)
        return 0;
    auto *base = reinterpret_cast<uint8_t *>(module);
    auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
    auto *nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress)
        return 0;
    int patched = 0;
    for (auto *imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress); imp->Name; imp++) {
        if (_stricmp(reinterpret_cast<const char *>(base + imp->Name), dll) != 0)
            continue;
        auto *names = reinterpret_cast<IMAGE_THUNK_DATA *>(
            base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto *addrs = reinterpret_cast<IMAGE_THUNK_DATA *>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; names++, addrs++) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal))
                continue;
            auto *ibn = reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char *>(ibn->Name), func) != 0)
                continue;
            DWORD old = 0;
            VirtualProtect(&addrs->u1.Function, sizeof(void *), PAGE_READWRITE, &old);
            addrs->u1.Function = reinterpret_cast<ULONG_PTR>(hook);
            VirtualProtect(&addrs->u1.Function, sizeof(void *), old, &old);
            patched++;
        }
    }
    return patched;
}

constexpr D3D12_RESOURCE_STATES kSlShaderRead =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *res, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    list->ResourceBarrier(1, &b);
}

D3D12_RESOURCE_DESC bufferDesc(UINT64 size)
{
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}

} // namespace

bool Dx12Fg::Impl::slStartRuntime()
{
    std::lock_guard<std::mutex> lock(g_sl_mutex);
    if (g_sl_tried)
        return g_sl_started; // once per process
    g_sl_tried = true;
    static std::wstring dir; // Streamline keeps the paths it is given
    dir = exeDir();
    g_sl_dll = LoadLibraryW((dir + L"\\sl.interposer.dll").c_str());
    if (!g_sl_dll) {
        log(1, "Streamline: sl.interposer.dll could not be loaded (error %lu)", GetLastError());
        return false;
    }
    static const sl::Feature features[] = { sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL };
    static const wchar_t *paths[1];
    paths[0] = dir.c_str();
    static std::wstring data;
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    data = std::wstring(tmp) + L"chiaki-ng-streamline";
    CreateDirectoryW(data.c_str(), nullptr);
    sl::Preferences pref;
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = data.c_str();
    // Manual hooking: Streamline hooks nothing, its proxies are asked for where they are needed.
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eDisableDebugText
        | sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = static_cast<uint32_t>(std::size(features));
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = kProjectId;
    pref.renderAPI = sl::RenderAPI::eD3D12;
    const sl::Result r = slInit(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) {
        log(1, "Streamline: slInit failed (%d)", static_cast<int>(r));
        return false;
    }
    g_sl_started = true;
    return true;
}

// Called right after the device is created: gives it to Streamline and, when Streamline offers
// DLSS Frame Generation on it, makes `queue` the native queue of a queue from its proxy device.
bool Dx12Fg::Impl::slAttachDevice()
{
    sl::Result r = slSetD3DDevice(device.Get());
    if (r != sl::Result::eOk) {
        log(1, "Streamline: slSetD3DDevice failed (%d)", static_cast<int>(r));
        return false;
    }
    sl::AdapterInfo adapter_info;
    adapter_info.deviceLUID = reinterpret_cast<uint8_t *>(&luid);
    adapter_info.deviceLUIDSizeInBytes = sizeof(luid);
    r = slIsFeatureSupported(sl::kFeatureDLSS_G, adapter_info);
    if (r != sl::Result::eOk) {
        sl::FeatureRequirements req;
        if (slGetFeatureRequirements(sl::kFeatureDLSS_G, req) == sl::Result::eOk)
            log(1, "Streamline: DLSS Frame Generation needs %sdriver %u.%u (this one %u.%u) and Windows build %u (this one %u)",
                (req.flags & sl::FeatureRequirementFlags::eHardwareSchedulingRequired)
                    ? "hardware-accelerated GPU scheduling, " : "",
                req.driverVersionRequired.major, req.driverVersionRequired.minor, req.driverVersionDetected.major,
                req.driverVersionDetected.minor, req.osVersionRequired.build, req.osVersionDetected.build);
        log(1, "Streamline: DLSS Frame Generation is not offered on this GPU (%d; RTX 20 / RTX 30 need MFG Unlocker)",
            static_cast<int>(r));
        return false;
    }
    ID3D12Device *proxy = device.Get();
    if (slUpgradeInterface(reinterpret_cast<void **>(&proxy)) != sl::Result::eOk || !proxy || proxy == device.Get()) {
        log(1, "Streamline: no proxy for the D3D12 device");
        return false;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(proxy->CreateCommandQueue(&qd, IID_PPV_ARGS(&sl_queue)))) {
        log(1, "Streamline: its proxy device could not create a queue");
        return false;
    }
    ID3D12CommandQueue *native = nullptr;
    if (slGetNativeInterface(sl_queue.Get(), reinterpret_cast<void **>(&native)) != sl::Result::eOk || !native) {
        log(1, "Streamline: no native queue behind its proxy queue");
        sl_queue.Reset();
        return false;
    }
    queue = native; // a reference of our own
    sl_device = proxy;
    sl_route = true;
    sl::DLSSGState state;
    if (slDLSSGGetState(sl::ViewportHandle(0), state, nullptr) == sl::Result::eOk && state.numFramesToGenerateMax > 0)
        sl_max_generated = std::min(static_cast<int>(state.numFramesToGenerateMax), static_cast<int>(kMaxOutputs));
    log(0, "Streamline: DLSS Frame Generation offered, up to %u generated frames per real frame (%d used)",
        state.numFramesToGenerateMax, sl_max_generated);
    return true;
}

// After init: the hooks, the frame generation window, the proxy factory and what the copies need.
bool Dx12Fg::Impl::slSetup()
{
    WNDCLASSW wc = {};
    wc.lpfnWndProc = hiddenWndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"chiaki_dx12fg_sl_window";
    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        log(1, "Streamline: could not register its window class");
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_sl_mutex);
        if (!g_sl_hooked) {
            // The factory hooks tell which queue Streamline presents the frame generation window
            // with. The Present hook goes on DXGI's swapchain class (one vtable for all of its
            // swapchains), reached through a throwaway swapchain before Streamline makes its own.
            void **fvt = *reinterpret_cast<void ***>(factory.Get());
            bool ok = patchSlot(fvt, 15, reinterpret_cast<void *>(&hookCreateForHwnd),
                                reinterpret_cast<void **>(&g_orig_create_hwnd))
                && patchSlot(fvt, 10, reinterpret_cast<void *>(&hookCreate), reinterpret_cast<void **>(&g_orig_create));
            HWND tmp = CreateWindowExW(0, wc.lpszClassName, L"chiaki DXGI hook", WS_POPUP, 0, 0, 64, 64, nullptr, nullptr,
                                       wc.hInstance, nullptr);
            ComPtr<IDXGIFactory2> factory2;
            ComPtr<IDXGISwapChain1> sc;
            DXGI_SWAP_CHAIN_DESC1 sd = {};
            sd.Width = 64;
            sd.Height = 64;
            sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            sd.SampleDesc.Count = 1;
            sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            sd.BufferCount = 2;
            sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            ok = ok && tmp && SUCCEEDED(factory.As(&factory2))
                && SUCCEEDED(factory2->CreateSwapChainForHwnd(queue.Get(), tmp, &sd, nullptr, nullptr, &sc));
            if (ok) {
                void **vt = *reinterpret_cast<void ***>(sc.Get());
                ok = patchSlot(vt, 8, reinterpret_cast<void *>(&hookPresent), reinterpret_cast<void **>(&g_orig_present))
                    && patchSlot(vt, 22, reinterpret_cast<void *>(&hookPresent1), reinterpret_cast<void **>(&g_orig_present1));
            }
            sc.Reset();
            if (tmp)
                DestroyWindow(tmp);
            g_sl_hooked = ok;
        }
        if (!g_sl_hooked) {
            log(1, "Streamline: DXGI's Present could not be hooked");
            return false;
        }
    }
    sl_hwnd = CreateWindowExW(0, wc.lpszClassName, L"chiaki DLSS Frame Generation (Streamline)", WS_OVERLAPPEDWINDOW, 0, 0,
                              kHiddenWindowSize, kHiddenWindowSize * 9 / 16, nullptr, nullptr, wc.hInstance, nullptr);
    if (!sl_hwnd) {
        log(1, "Streamline: could not create its hidden window");
        return false;
    }
    g_sl_hwnd.store(sl_hwnd);
    IDXGIFactory4 *proxy = factory.Get();
    if (slUpgradeInterface(reinterpret_cast<void **>(&proxy)) != sl::Result::eOk || !proxy || proxy == factory.Get()) {
        log(1, "Streamline: no proxy for the DXGI factory");
        return false;
    }
    sl_factory = proxy;
    ComPtr<IDXGIFactory5> factory5;
    BOOL tearing = FALSE;
    if (SUCCEEDED(factory.As(&factory5)))
        factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    sl_tearing = tearing != FALSE;

    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&sl_cap_fence))))
        return false;
    sl_cap_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (auto &cl : sl_cap_lists) {
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cl.allocator)))
            || FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cl.allocator.Get(), nullptr,
                                                IID_PPV_ARGS(&cl.list))))
            return false;
        cl.list->Close();
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC rd = bufferDesc(kSlReadback * kSlReadbackStride);
    void *mapped = nullptr;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&sl_readback)))
        || FAILED(sl_readback->Map(0, nullptr, &mapped)))
        return false;
    sl_readback_ptr = static_cast<const uint8_t *>(mapped);
    // A GPU timestamp with each copy tells when the frame was there: Streamline's present call
    // comes before the GPU has generated the frame.
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = static_cast<UINT>(kSlReadback);
    if (FAILED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&sl_ts_heap))))
        sl_ts_heap.Reset(); // then it is when the copy is seen done

    // Reflex has to be on while DLSS-G runs; no Reflex sleep, chiaki paces its frames itself.
    sl::ReflexOptions reflex;
    reflex.mode = sl::ReflexMode::eLowLatency;
    const sl::Result rr = slReflexSetOptions(reflex);
    if (rr != sl::Result::eOk)
        log(1, "Streamline: slReflexSetOptions failed (%d)", static_cast<int>(rr));
    g_sl_trace = std::getenv("CHIAKI_DLSSG_SL_TRACE") != nullptr;
    const char *always = std::getenv("CHIAKI_DLSSG_SL_ALWAYS");
    if (always && always[0] == '1') {
        std::lock_guard<std::mutex> lock(g_sl_mutex);
        if (!g_sl_focus_window) {
            g_sl_focus_window = CreateWindowExW(0, wc.lpszClassName, L"chiaki DLSS-G focus", WS_POPUP, 0, 0, 64, 64,
                                                nullptr, nullptr, wc.hInstance, nullptr);
            int patched = 0;
            for (const wchar_t *m : { L"sl.common.dll", L"sl.dlss_g.dll", L"sl.pcl.dll", L"sl.reflex.dll", L"sl.interposer.dll" })
                patched += patchImport(GetModuleHandleW(m), "user32.dll", "GetForegroundWindow",
                                       reinterpret_cast<void *>(&hookGetForegroundWindow));
            log(0, "Streamline: CHIAKI_DLSSG_SL_ALWAYS=1, frames are generated also while chiaki has no focus (%d imports)",
                patched);
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        g_sl_capture_owner = this;
    }
    log(0, "Streamline ready (tearing %s); its log is in %%TEMP%%\\chiaki-ng-streamline", sl_tearing ? "allowed" : "off");
    return true;
}

bool Dx12Fg::Impl::slSetOptions(int count)
{
    sl::DLSSGOptions o;
    o.mode = sl::DLSSGMode::eOn;
    o.numFramesToGenerate = static_cast<uint32_t>(count);
    o.numBackBuffers = static_cast<uint32_t>(kSlBackBuffers);
    o.colorWidth = o.mvecDepthWidth = static_cast<uint32_t>(width);
    o.colorHeight = o.mvecDepthHeight = static_cast<uint32_t>(height + 1);
    o.colorBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    o.mvecBufferFormat = DXGI_FORMAT_R16G16_FLOAT;
    o.depthBufferFormat = DXGI_FORMAT_R32_FLOAT;
    const sl::Result r = slDLSSGSetOptions(sl::ViewportHandle(0), o);
    if (r != sl::Result::eOk) {
        log(1, "Streamline: slDLSSGSetOptions failed (%d)", static_cast<int>(r));
        return false;
    }
    sl_count_set = count;
    return true;
}

// Frame copies not handed out yet are dropped; the ones Vulkan may still read stay taken.
void Dx12Fg::Impl::slResetSortingLocked()
{
    for (auto &s : sl_ring) {
        if (s.state == kSlCaptured)
            s.state = kSlFree;
    }
    sl_records.clear();
    sl_gens.clear();
    sl_ready.clear();
    sl_ready_valid = false;
    sl_waiting = false;
    sl_last_real = -1;
}

bool Dx12Fg::Impl::slEnsureSize(int w, int h)
{
    stage = "Streamline buffers";
    // The frame in the shared buffers: w x h RGBA8, rows as D3D12 lays them out (no texture).
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = static_cast<UINT64>(w);
    td.Height = static_cast<UINT>(h);
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    device->GetCopyableFootprints(&td, 0, 1, 0, &color.footprint, nullptr, nullptr, &color.total);
    color.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color.bytes_per_texel = 4;
    // Depth and motion as DLSS-G wants them, constant: mid-range depth, zero motion (it finds
    // the motion itself with its optical flow).
    bool ok = makeShared(color_buf, color.total) && makeTex(depth, w, h + 1, DXGI_FORMAT_R32_FLOAT, 4)
        && makeTex(motion, w, h + 1, DXGI_FORMAT_R16G16_FLOAT, 4);
    for (auto &s : sl_ring)
        ok = ok && makeShared(s.buf, color.total);
    ok = ok && fill({ &motion }, &depth);
    if (ok) {
        sl_marker_stride = ((static_cast<UINT64>(w) * 4 + kSlReadbackStride - 1) / kSlReadbackStride) * kSlReadbackStride;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        const D3D12_RESOURCE_DESC rd = bufferDesc(sl_marker_stride * kSlots);
        void *mapped = nullptr;
        ok = SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                       nullptr, IID_PPV_ARGS(&sl_marker)))
            && SUCCEEDED(sl_marker->Map(0, nullptr, &mapped));
        if (ok) {
            sl_marker_ptr = static_cast<uint8_t *>(mapped);
            std::memset(sl_marker_ptr, 0, static_cast<size_t>(sl_marker_stride * kSlots));
        }
    }
    if (!ok)
        return false;

    // The swapchain: the frame with the marker row under it.
    stage = "Streamline swapchain";
    const UINT sw = static_cast<UINT>(w), sh = static_cast<UINT>(h + 1);
    SetWindowPos(sl_hwnd, nullptr, 0, 0, static_cast<int>(sw), static_cast<int>(sh),
                 SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = sw;
    sd.Height = sh;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = static_cast<UINT>(kSlBackBuffers);
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = sl_tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    g_sl_swapchain_hooked.store(-1);
    ComPtr<IDXGISwapChain1> sc1;
    const HRESULT hr = sl_factory->CreateSwapChainForHwnd(sl_queue.Get(), sl_hwnd, &sd, nullptr, nullptr, &sc1);
    if (FAILED(hr) || FAILED(sc1.As(&sl_swapchain))) {
        log(1, "Streamline: the frame generation swapchain could not be created (0x%08x)", static_cast<unsigned>(hr));
        return false;
    }
    factory->MakeWindowAssociation(sl_hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
    if (g_sl_swapchain_hooked.load() == 0) {
        log(1, "Streamline's swapchain is not one of DXGI's usual swapchains: its frames cannot be copied back");
        return false;
    }
    if (!g_sl_present_queue.load())
        log(1, "Streamline: the queue it presents with was not seen yet");
    for (size_t i = 0; i < kSlBackBuffers; i++) {
        if (FAILED(sl_swapchain->GetBuffer(static_cast<UINT>(i), IID_PPV_ARGS(&sl_buffers[i])))) {
            log(1, "Streamline: the swapchain's buffers are not available");
            return false;
        }
    }
    if (!slSetOptions(1))
        return false;
    sl::DLSSGState state;
    if (slDLSSGGetState(sl::ViewportHandle(0), state, nullptr) == sl::Result::eOk) {
        if (state.numFramesToGenerateMax > 0)
            sl_max_generated = max_generated =
                std::min(static_cast<int>(state.numFramesToGenerateMax), static_cast<int>(kMaxOutputs));
        log(0, "Streamline: DLSS Frame Generation on %ux%u (frame + marker row), up to %d generated frames, about %.0f MB",
            sw, sh, max_generated, static_cast<double>(state.estimatedVRAMUsageInBytes) / 1048576.0);
    }
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        slResetSortingLocked();
        sl_capture_on = true;
    }
    // Streamline sets DLSS-G up with the first frames it gets: a few now (whatever the buffers
    // hold), dropped, rather than a stall at the start of the stream.
    stage = "Streamline warm-up";
    for (int i = 0; i < 4 && ok; i++) {
        sl_frame = (sl_frame + 1) & kSlIdMask;
        ok = slPresentFrame(1, i == 0, false, 0);
        Sleep(16);
    }
    Sleep(100);
    ok = waitIdle() && ok;
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        slSortLocked();
        slResetSortingLocked();
        // What the GPU keeps up with is measured again at this size.
        sl_cap = max_generated;
        sl_first_ms = sl_spacing_ms = 0.0;
        sl_late_bits = 0;
        sl_room_frames = 0;
        sl_settle = 30;
        sl_probe_wait = 600;
        sl_probing = false;
    }
    if (!ok)
        log(1, "Streamline: the warm-up frames failed");
    return ok;
}

void Dx12Fg::Impl::slDestroySized()
{
    uint64_t recorded = 0;
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        sl_capture_on = false;
        recorded = sl_cap_seq;
    }
    // The copies already recorded finish before their buffers go.
    for (int i = 0; i < 200 && sl_cap_fence && sl_cap_fence->GetCompletedValue() < recorded; i++)
        Sleep(10);
    if (sl_swapchain) {
        sl::DLSSGOptions off;
        off.mode = sl::DLSSGMode::eOff;
        slDLSSGSetOptions(sl::ViewportHandle(0), off);
    }
    sl_count_set = 0;
    for (auto &b : sl_buffers)
        b.Reset();
    sl_swapchain.Reset();
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        slResetSortingLocked();
        for (auto &s : sl_ring) {
            freeShared(s.buf);
            s.state = kSlFree;
            s.free_after = 0;
        }
    }
    if (sl_marker)
        sl_marker->Unmap(0, nullptr);
    sl_marker.Reset();
    sl_marker_ptr = nullptr;
}

// One real frame into Streamline: frame token, constants, the frame copied into the back buffer
// with the marker row under it (the frame id in RGB, alpha 0), the tags, Present. With `signal`
// the shared timeline gets `signal_value` once the copy is queued, on every path.
bool Dx12Fg::Impl::slPresentFrame(int count, bool reset, bool signal, uint64_t signal_value)
{
    const sl::ViewportHandle viewport(0);
    bool ok = true;
    if (count != sl_count_set) {
        stage = "slDLSSGSetOptions";
        ok = slSetOptions(count);
    }
    sl::FrameToken *token = nullptr;
    if (ok) {
        stage = "slGetNewFrameToken";
        const uint32_t index = sl_frame;
        ok = slGetNewFrameToken(token, &index) == sl::Result::eOk && token;
    }
    if (ok) {
        slPCLSetMarker(sl::PCLMarker::eSimulationStart, *token);
        slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *token);
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *token);
        // A stationary camera, as for NGX (video carries no camera).
        sl::Constants c = {};
        sl::float4x4 identity;
        identity.setRow(0, { 1, 0, 0, 0 });
        identity.setRow(1, { 0, 1, 0, 0 });
        identity.setRow(2, { 0, 0, 1, 0 });
        identity.setRow(3, { 0, 0, 0, 1 });
        c.cameraViewToClip = c.clipToCameraView = c.clipToLensClip = c.clipToPrevClip = c.prevClipToClip = identity;
        c.jitterOffset = { 0, 0 };
        c.mvecScale = { 1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height + 1) };
        c.cameraPinholeOffset = { 0, 0 };
        c.cameraPos = { 0, 0, 0 };
        c.cameraUp = { 0, 1, 0 };
        c.cameraRight = { 1, 0, 0 };
        c.cameraFwd = { 0, 0, 1 };
        c.cameraNear = 0.1f;
        c.cameraFar = 1000.0f;
        c.cameraFOV = 1.0f;
        c.cameraAspectRatio = static_cast<float>(width) / static_cast<float>(height + 1);
        c.depthInverted = sl::Boolean::eFalse;
        c.cameraMotionIncluded = sl::Boolean::eTrue;
        c.motionVectors3D = sl::Boolean::eFalse;
        c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        c.orthographicProjection = sl::Boolean::eFalse;
        c.motionVectorsDilated = sl::Boolean::eFalse;
        c.motionVectorsJittered = sl::Boolean::eFalse;
        stage = "slSetConstants";
        ok = slSetConstants(c, *token, viewport) == sl::Result::eOk;
    }
    Slot *slot = nullptr;
    const bool begun = ok && beginList(&slot);
    ok = ok && begun;
    const D3D12_RESOURCE_STATES saved_depth = depth.state, saved_motion = motion.state;
    if (ok) {
        const size_t slot_index = static_cast<size_t>(slot - slots.data());
        uint8_t *marker = sl_marker_ptr + slot_index * sl_marker_stride;
        marker[0] = static_cast<uint8_t>(sl_frame & 0xff);
        marker[1] = static_cast<uint8_t>((sl_frame >> 8) & 0xff);
        marker[2] = static_cast<uint8_t>((sl_frame >> 16) & 0xff);
        marker[3] = 0;
        stage = "copy into Streamline's back buffer";
        // Through the proxy: DLSS-G wants GetCurrentBackBufferIndex called.
        ID3D12Resource *bb = sl_buffers[sl_swapchain->GetCurrentBackBufferIndex() % kSlBackBuffers].Get();
        transition(list.Get(), bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = bb;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = color_buf.res.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = color.footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        src.pResource = sl_marker.Get();
        src.PlacedFootprint.Offset = slot_index * sl_marker_stride;
        src.PlacedFootprint.Footprint.Height = 1;
        list->CopyTextureRegion(&dst, 0, static_cast<UINT>(height), 0, &src, nullptr);
        transition(list.Get(), bb, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
        barrier(depth, kSlShaderRead);
        barrier(motion, kSlShaderRead);
        sl::Resource depth_res(sl::ResourceType::eTex2d, depth.res.Get(), nullptr, nullptr, kSlShaderRead);
        sl::Resource motion_res(sl::ResourceType::eTex2d, motion.res.Get(), nullptr, nullptr, kSlShaderRead);
        sl::Extent extent{ 0, 0, static_cast<uint32_t>(width), static_cast<uint32_t>(height + 1) };
        sl::ResourceTag tags[] = {
            sl::ResourceTag(&depth_res, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow, &extent),
            sl::ResourceTag(&motion_res, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow, &extent),
        };
        stage = "slSetTagForFrame";
        ok = slSetTagForFrame(*token, viewport, tags, 2, reinterpret_cast<sl::CommandBuffer *>(list.Get()))
            == sl::Result::eOk;
    }
    if (ok) {
        stage = "ExecuteCommandLists";
        ok = endList(*slot);
    } else if (begun) {
        // Nothing of this list runs.
        list->Close();
        depth.state = saved_depth;
        motion.state = saved_motion;
    }
    if (signal && !signalShared(signal_value))
        ok = false;
    if (ok) {
        slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *token);
        slPCLSetMarker(sl::PCLMarker::ePresentStart, *token);
        stage = "Present (Streamline)";
        const HRESULT hr = sl_swapchain->Present(0, sl_tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
        slPCLSetMarker(sl::PCLMarker::ePresentEnd, *token);
        if (FAILED(hr)) {
            log(1, "Streamline: Present failed (0x%08x)", static_cast<unsigned>(hr));
            ok = false;
        }
    }
    return ok;
}

bool Dx12Fg::Impl::slGenerate(int count, uint64_t wait_value, uint64_t signal_value, bool reset)
{
    count = std::clamp(count, 1, max_generated);
    const double now = nowMs();
    if (sl_t_generate > 0.0) {
        const double dt = now - sl_t_generate;
        if (dt > 2.0 && dt < 100.0)
            sl_interval += (dt - sl_interval) * 0.1;
    }
    {
        std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
        // The frame before's generated frames that nobody took are not wanted anymore.
        for (int s : sl_ready)
            slFreeLocked(s);
        sl_ready.clear();
        sl_ready_valid = false;
        sl_frame = (sl_frame + 1) & kSlIdMask;
        sl_count = count;
        sl_waiting = true;
        sl_t_generate = now;
    }
    stage = "queue wait";
    // GPU side wait for the Vulkan copy into the shared buffer.
    if (!sized || !sl_swapchain || FAILED(queue->Wait(shared_fence.Get(), wait_value))) {
        signalShared(signal_value);
        return false;
    }
    return slPresentFrame(count, reset, true, signal_value);
}

// The Present hook, with g_sl_capture_mutex held: copies the frame being presented on
// Streamline's window into a free slot and its marker pixel into the readback ring, on
// Streamline's queue, ahead of the present.
uint64_t Dx12Fg::Impl::slCapture(IDXGISwapChain *sc, UINT sync, UINT flags)
{
    if (!sl_capture_on)
        return 0;
    ID3D12CommandQueue *present_queue = g_sl_present_queue.load();
    ComPtr<IDXGISwapChain3> sc3;
    ComPtr<ID3D12Resource> bb;
    if (!present_queue || FAILED(sc->QueryInterface(IID_PPV_ARGS(&sc3)))
        || FAILED(sc3->GetBuffer(sc3->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&bb)))) {
        sl_stat_skipped++;
        return 0;
    }
    const D3D12_RESOURCE_DESC bd = bb->GetDesc();
    if (bd.Width != static_cast<UINT64>(width) || bd.Height != static_cast<UINT>(height + 1)) {
        sl_stat_skipped++; // a frame of the size before
        return 0;
    }
    if (bd.Format != DXGI_FORMAT_R8G8B8A8_UNORM && bd.Format != DXGI_FORMAT_R8G8B8A8_TYPELESS
        && bd.Format != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
        if (!sl_format_warned) {
            sl_format_warned = true;
            log(1, "Streamline presents in format %d, not RGBA8: its frames cannot be copied back", static_cast<int>(bd.Format));
        }
        sl_stat_skipped++;
        return 0;
    }
    const uint64_t seq = sl_cap_seq;
    SlCapList &cl = sl_cap_lists[seq % kSlCapLists];
    if (cl.value && sl_cap_fence->GetCompletedValue() < cl.value) {
        if (FAILED(sl_cap_fence->SetEventOnCompletion(cl.value, sl_cap_event))
            || WaitForSingleObject(sl_cap_event, 100) != WAIT_OBJECT_0) {
            sl_stat_skipped++;
            return 0;
        }
    }
    if (FAILED(cl.allocator->Reset()) || FAILED(cl.list->Reset(cl.allocator.Get(), nullptr))) {
        sl_stat_skipped++;
        return 0;
    }
    int slot = -1;
    const uint64_t shared_done = shared_fence->GetCompletedValue();
    for (size_t i = 0; i < kSlRing && slot < 0; i++) {
        const size_t k = (sl_slot_next + i) % kSlRing;
        const SlSlot &s = sl_ring[k];
        if (s.state == kSlFree || (s.state == kSlTaken && shared_done >= s.free_after))
            slot = static_cast<int>(k);
    }
    if (slot >= 0) {
        sl_slot_next = (static_cast<size_t>(slot) + 1) % kSlRing;
        sl_ring[static_cast<size_t>(slot)].state = kSlCaptured;
    } else {
        sl_stat_dropped++;
    }
    ID3D12GraphicsCommandList *l = cl.list.Get();
    transition(l, bb.Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = bb.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    if (slot >= 0) {
        dst.pResource = sl_ring[static_cast<size_t>(slot)].buf.res.Get();
        dst.PlacedFootprint = color.footprint;
        dst.PlacedFootprint.Footprint.Format = bd.Format;
        const D3D12_BOX frame = { 0, 0, 0, static_cast<UINT>(width), static_cast<UINT>(height), 1 };
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, &frame);
    }
    dst.pResource = sl_readback.Get();
    dst.PlacedFootprint = {};
    dst.PlacedFootprint.Offset = (seq % kSlReadback) * kSlReadbackStride;
    dst.PlacedFootprint.Footprint = { bd.Format, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT };
    const D3D12_BOX marker = { 0, static_cast<UINT>(height), 0, 1, static_cast<UINT>(height + 1), 1 };
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, &marker);
    transition(l, bb.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    if (sl_ts_heap) {
        if (!sl_ts_freq && FAILED(present_queue->GetTimestampFrequency(&sl_ts_freq)))
            sl_ts_freq = 0;
        if (sl_ts_freq && nowMs() - sl_ts_calibrated > 1000.0
            && SUCCEEDED(present_queue->GetClockCalibration(&sl_ts_gpu0, &sl_ts_cpu0)))
            sl_ts_calibrated = nowMs();
        const UINT q = static_cast<UINT>(seq % kSlReadback);
        l->EndQuery(sl_ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q);
        l->ResolveQueryData(sl_ts_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, q, 1, sl_readback.Get(),
                            q * kSlReadbackStride + kSlTimestampOffset);
    }
    if (FAILED(l->Close())) {
        if (slot >= 0)
            sl_ring[static_cast<size_t>(slot)].state = kSlFree;
        sl_stat_skipped++;
        return 0;
    }
    ID3D12CommandList *lists[] = { l };
    present_queue->ExecuteCommandLists(1, lists);
    cl.value = seq + 1;
    present_queue->Signal(sl_cap_fence.Get(), cl.value);
    sl_cap_seq = seq + 1;
    sl_records.push_back({ seq, slot });
    sl_cap_time[seq % kSlReadback] = nowMs();
    if (!g_sl_trace)
        return 0;
    sl_trace_t[seq % kSlReadback] = nowMs() - sl_t_generate;
    sl_trace_present[seq % kSlReadback] = -1.0f;
    sl_trace_mode[seq % kSlReadback] = (sync << 16) | (flags & 0xffff);
    return seq + 1;
}

void Dx12Fg::Impl::slFreeLocked(int slot)
{
    if (slot >= 0 && sl_ring[static_cast<size_t>(slot)].state == kSlCaptured)
        sl_ring[static_cast<size_t>(slot)].state = kSlFree;
}

// Sorts the copies the GPU has finished, in presentation order: a marker with alpha 0 is a real
// frame (its id in RGB), anything else a generated one. The generated frames presented before
// real frame N lead up to N.
void Dx12Fg::Impl::slSortLocked()
{
    if (!sl_cap_fence)
        return;
    const uint64_t done = sl_cap_fence->GetCompletedValue();
    const double seen = nowMs();
    while (!sl_records.empty() && sl_records.front().seq < done) {
        const SlRecord rec = sl_records.front();
        sl_records.pop_front();
        if (sl_cap_seq - rec.seq >= kSlReadback) {
            slFreeLocked(rec.slot); // its marker pixel has been overwritten
            continue;
        }
        const uint8_t *px = sl_readback_ptr + (rec.seq % kSlReadback) * kSlReadbackStride;
        if (g_sl_trace && sl_trace_lines < 400) {
            const size_t k = rec.seq % kSlReadback;
            char item[96];
            snprintf(item, sizeof(item), " %s@%.1f(p%.1f s%u f%x)", px[3] != 0 ? "G" : "R", sl_trace_t[k],
                     static_cast<double>(sl_trace_present[k]), sl_trace_mode[k] >> 16, sl_trace_mode[k] & 0xffff);
            sl_trace_line += item;
        }
        if (px[3] != 0) {
            const double t = sl_cap_time[rec.seq % kSlReadback];
            // When it was there: its GPU timestamp (between the present call and now), or now.
            double there = seen;
            if (sl_ts_heap && sl_ts_freq) {
                uint64_t ts = 0;
                memcpy(&ts, px + kSlTimestampOffset, sizeof(ts));
                const double gpu = qpcMs(sl_ts_cpu0)
                    + (static_cast<double>(ts) - static_cast<double>(sl_ts_gpu0)) * 1000.0 / static_cast<double>(sl_ts_freq);
                if (ts && gpu > t - 1.0 && gpu <= seen)
                    there = gpu;
            }
            if (sl_gens.empty()) {
                sl_gens_times.first_cap = t;
                sl_gens_times.first_seen = there;
            }
            sl_gens_times.last_cap = t;
            sl_gens_times.last_seen = there;
            sl_gens.push_back(rec.slot);
            continue;
        }
        slFreeLocked(rec.slot); // a real frame: chiaki has it already
        const int64_t id = static_cast<int64_t>(px[0]) | (static_cast<int64_t>(px[1]) << 8)
            | (static_cast<int64_t>(px[2]) << 16);
        if (sl_waiting && id == static_cast<int64_t>(sl_frame) && !sl_ready_valid) {
            sl_ready.swap(sl_gens);
            sl_ready_valid = true;
            sl_ready_times = sl_gens_times;
        }
        for (int s : sl_gens)
            slFreeLocked(s);
        sl_gens.clear();
        sl_last_real = id;
    }
}

bool Dx12Fg::Impl::slFramesReady()
{
    std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
    if (!sl_waiting)
        return true;
    slSortLocked();
    if (sl_ready_valid)
        return true;
    const int64_t previous = static_cast<int64_t>((sl_frame - 1) & kSlIdMask);
    if (sl_last_real == previous && static_cast<int>(sl_gens.size()) >= sl_count)
        return true;
    return nowMs() - sl_t_generate > slTimeoutMs();
}

// After each taken frame: whether all its generated frames were there before the next stream frame
// was due. When most of the recent frames were late, one fewer is offered; after a while with
// room for one more (from the times measured), one more.
void Dx12Fg::Impl::slAdaptLocked(int n, const SlTimes &times)
{
    const double interval = slFrameIntervalMs();
    const double budget = interval - 1.5; // the copy back and chiaki's own work after it
    if (n >= 1) {
        const double first = times.first_seen - sl_t_generate;
        sl_first_ms = sl_first_ms > 0.0 ? sl_first_ms + (first - sl_first_ms) * 0.1 : first;
    }
    if (n >= 2) {
        // From Streamline's present calls: one every frame's GPU time (or its pacing).
        const double spacing = (times.last_cap - times.first_cap) / (n - 1);
        sl_spacing_ms = sl_spacing_ms > 0.0 ? sl_spacing_ms + (spacing - sl_spacing_ms) * 0.1 : spacing;
    }
    const bool late = n < sl_count || times.last_seen - sl_t_generate > budget;
    const int was = sl_cap;
    if (g_sl_trace && sl_trace_lines < 400) {
        char item[160];
        snprintf(item, sizeof(item), " | n%d/%d seen %.1f..%.1f cap %.1f..%.1f late %d settle %d bits %x", n, sl_count,
                 times.first_seen - sl_t_generate, times.last_seen - sl_t_generate, times.first_cap - sl_t_generate,
                 times.last_cap - sl_t_generate, late ? 1 : 0, sl_settle, sl_late_bits & 0xffffu);
        sl_trace_line += item;
    }
    if (sl_settle > 0) {
        // After a change the frames still queued in DLSS-G come out late; not counted.
        sl_settle--;
        return;
    }
    sl_late_bits = (sl_late_bits << 1) | (late ? 1u : 0u);
    int recent_late = 0;
    for (uint32_t b = sl_late_bits & 0xffffu; b; b &= b - 1)
        recent_late++;
    if (sl_probing && ++sl_probe_frames >= 300)
        sl_probing = false; // the one more held
    bool probe = false;
    if (recent_late >= 8 && sl_count > 1) {
        // One fewer at a time: while DLSS-G is behind, the times measured are too long to go by.
        sl_cap = sl_count - 1;
        if (sl_probing)
            sl_probe_wait = std::min(sl_probe_wait * 2, 18000); // tried too soon
        sl_probing = false;
        sl_late_bits = 0;
        sl_room_frames = 0;
        sl_settle = 30;
    } else if (!late && sl_count == sl_cap && sl_cap < max_generated) {
        // One more: soon when the times measured say it fits, otherwise now and then as a try
        // (the times can be from a slower moment, GPU clocks still low for example).
        const bool fits = sl_spacing_ms > 0.0 && sl_first_ms + sl_cap * sl_spacing_ms < budget - 1.5;
        if (++sl_room_frames >= (fits ? 120 : sl_probe_wait)) {
            sl_cap++;
            probe = !fits;
            sl_probing = probe;
            sl_probe_frames = 0;
            sl_late_bits = 0;
            sl_room_frames = 0;
            sl_settle = 30;
        }
    } else {
        sl_room_frames = 0;
    }
    if (probe)
        log(0, "Streamline: trying %d generated frames per stream frame at %dx%d again", sl_cap, width, height);
    else if (sl_cap != was)
        log(0, "Streamline: the GPU keeps up with %d generated frame%s per stream frame at %dx%d (the first "
               "after %.1f ms, then one every %.1f ms, %.1f ms between stream frames), was %d",
            sl_cap, sl_cap == 1 ? "" : "s", width, height, sl_first_ms, sl_spacing_ms, interval, was);
}

int Dx12Fg::Impl::slTakeFrames()
{
    std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
    if (!sl_waiting)
        return 0;
    slSortLocked();
    sl_waiting = false;
    const int64_t previous = static_cast<int64_t>((sl_frame - 1) & kSlIdMask);
    std::vector<int> *src = sl_ready_valid ? &sl_ready : (sl_last_real == previous ? &sl_gens : nullptr);
    const bool complete = sl_ready_valid || (src && static_cast<int>(src->size()) >= sl_count);
    const SlTimes times = sl_ready_valid ? sl_ready_times : sl_gens_times;
    int n = 0;
    if (src) {
        for (int s : *src) {
            if (s >= 0 && n < sl_count) {
                SlSlot &taken = sl_ring[static_cast<size_t>(s)];
                taken.state = kSlTaken;
                // Every value after the current one on the shared timeline comes after the Vulkan
                // copy out of this slot (it is submitted next).
                taken.free_after = last_value + 1;
                sl_taken[static_cast<size_t>(n++)] = s;
            } else {
                slFreeLocked(s);
            }
        }
        src->clear();
    }
    sl_ready_valid = false;
    // A real frame presented without any generated one: DLSS-G is paused (no window of chiaki
    // has the focus) or started over; that says nothing about what the GPU keeps up with.
    if (n > 0 || !complete)
        slAdaptLocked(n, times);

    const double now = nowMs();
    const double waited = now - sl_t_generate;
    sl_stat_frames++;
    sl_stat_gens += static_cast<uint64_t>(n);
    if (n < sl_count)
        sl_stat_short++;
    if (!complete)
        sl_stat_timeouts++;
    sl_stat_wait_ms += waited;
    if (g_sl_trace && sl_trace_lines < 400) {
        sl_trace_lines++;
        log(0, "Streamline trace, frame %u taken at %.1f ms:%s", sl_frame, waited, sl_trace_line.c_str());
        sl_trace_line.clear();
    }
    if (sl_traced < 4) {
        sl_traced++;
        log(0, "Streamline: frame %u, %d of %d generated frames back after %.1f ms", sl_frame, n, sl_count, waited);
    }
    if (sl_stat_logged == 0.0)
        sl_stat_logged = now;
    if (now - sl_stat_logged > 10000.0) {
        log(0, "Streamline: %llu frames, %.2f generated frames each (%d asked), %llu short, %llu not complete in time, "
               "%.1f ms until they were in; copies dropped %llu, skipped %llu",
            static_cast<unsigned long long>(sl_stat_frames), static_cast<double>(sl_stat_gens) / sl_stat_frames, sl_count,
            static_cast<unsigned long long>(sl_stat_short), static_cast<unsigned long long>(sl_stat_timeouts),
            sl_stat_wait_ms / sl_stat_frames, static_cast<unsigned long long>(sl_stat_dropped),
            static_cast<unsigned long long>(sl_stat_skipped));
        sl_stat_frames = sl_stat_gens = sl_stat_short = sl_stat_timeouts = sl_stat_dropped = sl_stat_skipped = 0;
        sl_stat_wait_ms = 0.0;
        sl_stat_logged = now;
    }
    return n;
}
#endif // CHIAKI_HAVE_STREAMLINE

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
    // Not when another window of the process presents on the provider's device already.
    if (!mayPresent())
        return true;
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
    g_ngx_d3d12_up.fetch_add(1);
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
    if (sl_route)
        ; // frame generation goes through Streamline
    else if (fg_available)
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
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (sl_route)
        slDestroySized();
#endif
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

// Feature 18's controls of one pass, set before its CreateFeature and every evaluation. Skin
// structure only takes effect with the automatic mask (-1 = the model's own, as 1). Values out
// of range are not clamped by the snippet: it falls back to the default (intensity 2 = 1).
void Dx12Fg::Impl::nrSetKeys(int pass)
{
    const NrTuning &t = nr_tuning[static_cast<size_t>(pass)];
    NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Style", static_cast<unsigned int>(t.style));
    NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.Intensity", t.intensity);
    NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.LocalStructureStrength", t.local_structure);
    NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.LocalToneStrength", t.local_tone);
    NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.UseAutoMask", t.auto_mask ? 1u : 0u);
    NVSDK_NGX_Parameter_SetF(nr_params, "DLSSNR.SkinStructureStrength", t.auto_mask ? t.skin_structure : -1.0f);
    NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.UICorrection", 1);
#if defined(CHIAKI_DX12FG_TEST_HOOKS)
    // Test hook (dev builds only): CHIAKI_DX12FG_NR_KEYS="DLSSNR.Name=u3;DLSSNR.Other=f0.5"
    // sets more keys (u = unsigned, f = float) after these.
    size_t at = 0;
    while (at < nr_test_keys.size()) {
        size_t end = nr_test_keys.find(';', at);
        if (end == std::string::npos)
            end = nr_test_keys.size();
        const std::string item = nr_test_keys.substr(at, end - at);
        const size_t eq = item.find('=');
        if (eq != std::string::npos && eq + 2 <= item.size()) {
            const std::string key = item.substr(0, eq);
            const char kind = item[eq + 1];
            const std::string value = item.substr(eq + 2);
            if (kind == 'u')
                NVSDK_NGX_Parameter_SetUI(nr_params, key.c_str(), static_cast<unsigned int>(std::stoul(value)));
            else if (kind == 'f')
                NVSDK_NGX_Parameter_SetF(nr_params, key.c_str(), std::stof(value));
            else if (kind == 'i')
                NVSDK_NGX_Parameter_SetI(nr_params, key.c_str(), std::stoi(value));
        }
        at = end + 1;
    }
#endif
}

#if defined(CHIAKI_DX12FG_TEST_HOOKS)
// Reads CHIAKI_DX12FG_NR_KEYS; true when it changed since the last call.
bool Dx12Fg::Impl::nrTestKeys()
{
    const char *env = std::getenv("CHIAKI_DX12FG_NR_KEYS");
    const std::string keys = env ? env : "";
    if (keys == nr_test_keys)
        return false;
    log(0, "test keys: \"%s\"", keys.c_str());
    nr_test_keys = keys;
    return true;
}
#endif

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
#if defined(CHIAKI_DX12FG_TEST_HOOKS)
        nrTestKeys();
#endif
        bool executed = true;
        for (int pass = 0; pass < nr_cfg.passes; pass++) {
            NVSDK_NGX_Handle *&handle = pass == 0 ? nr_handle : nr_more[static_cast<size_t>(pass - 1)];
            nrSetKeys(pass);
            Slot *slot = nullptr;
            if (!beginList(&slot))
                return false;
            stage = "nvngx_dlssnr CreateFeature";
            const auto r = static_cast<NVSDK_NGX_Result>(
                gate.create(list.Get(), kFeatureNeuralRendering, nr_params, reinterpret_cast<void **>(&handle)));
            executed = endList(*slot) && waitIdle() && executed;
            if (NVSDK_NGX_FAILED(r) || !handle) {
                log(1, "creating DLSS 5 neural rendering (feature 18, pass %d) failed: %s (0x%08x) %s", pass + 1,
                    ngxResultString(r), static_cast<unsigned>(r), gate.fault());
                handle = nullptr;
                return false;
            }
        }
        nr_tuning_set = nr_tuning;
        nr_tuning_valid = true;
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
    if (cs_encode && cs_compose && cs_scrgb && cs_heap)
        return true;
    if (cs_failed)
        return false;
    cs_failed = true; // until everything below succeeds
    stage = "HDR shaders";
    HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    const auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    using SerializeFn = HRESULT(WINAPI *)(const D3D12_ROOT_SIGNATURE_DESC *, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob **,
                                          ID3DBlob **);
    const auto serialize = reinterpret_cast<SerializeFn>(GetProcAddress(d3d12_dll, "D3D12SerializeRootSignature"));
    if (!compile || !serialize) {
        log(1, "the HDR passes (NVIDIA's proxy, NeuralHub's panel window) need d3dcompiler_47.dll and "
               "D3D12SerializeRootSignature");
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
    for (int i = 0; i < 3; i++) {
        const char *entry = i == 0 ? "encode" : i == 1 ? "compose" : "scrgb";
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
        ComPtr<ID3D12PipelineState> &target = i == 0 ? cs_encode : i == 1 ? cs_compose : cs_scrgb;
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&target)))) {
            log(1, "could not create the HDR proxy pipeline %s", entry);
            return false;
        }
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 12; // 0-3 encode, 4-7 compose, 8-11 scrgb
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

// ---- NeuralHub's in-game panel ----

bool Dx12Fg::Impl::panelResize(UINT w, UINT h, DXGI_FORMAT format)
{
    if (w == panel_w && h == panel_h && format == panel_format)
        return true;
    // Nothing may still use the buffers (a Present, NeuralHub drawing its panel). Two
    // buffers: at the output size they are large (4K HDR: 66 MB each).
    waitIdle();
    const HRESULT hr = swapchain->ResizeBuffers(2, w, h, format, 0);
    if (FAILED(hr)) {
        log(1, "NeuralHub's panel window: the swapchain could not take %ux%u (0x%08x), the panel is not shown",
            w, h, static_cast<unsigned>(hr));
        return false;
    }
    panel_w = w;
    panel_h = h;
    panel_format = format;
    ComPtr<IDXGISwapChain3> sc3;
    if (SUCCEEDED(swapchain.As(&sc3))) {
        const DXGI_COLOR_SPACE_TYPE space = format == DXGI_FORMAT_R16G16B16A16_FLOAT
                                                ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                                                : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        UINT support = 0;
        if (SUCCEEDED(sc3->CheckColorSpaceSupport(space, &support))
            && (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
            sc3->SetColorSpace1(space);
    }
    return true;
}

// Before every Present of the hidden swapchain: false (from then on) once NeuralHub has hooked
// another window of this process. Called on the thread that presents; chiaki presents its own
// window on the same render thread, so nothing presents between the check and the Present.
bool Dx12Fg::Impl::mayPresent()
{
    if (shared_present)
        return false;
    const HWND other = providerHookedWindow(hwnd);
    if (!other)
        return true;
    shared_present = true;
    panelHide();
    // Hooked by an earlier Present of ours: back to our own procedure. The provider holds the
    // other window's procedure now and never touches ours again (we no longer present).
    if (hwnd && moduleOf(GetWindowLongPtrW(hwnd, GWLP_WNDPROC)) == GetModuleHandleW(L"neuralhub_host.dll"))
        SetWindowLongPtrW(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(hiddenWndProc));
    log(0, "window %p of this process presents on NeuralHub's device (chiaki's own window): the hidden swapchain is "
           "not presented, NeuralHub's runtime opens on that window and draws its in-game panel (End) on it",
        static_cast<void *>(other));
    return false;
}

void Dx12Fg::Impl::panelHide()
{
    panel_copy = false;
    if (panel_shown && hwnd)
        ShowWindow(hwnd, SW_HIDE);
    panel_shown = false;
}

// Before the evaluation's command list: follows the log, and decides whether this frame goes
// into the swapchain (the panel is open and chiaki's window has the focus).
void Dx12Fg::Impl::panelUpdate()
{
    panel_copy = false;
    if (!panelRoute())
        return;
    if (!panel_log_tried) {
        panel_log_tried = true;
        const std::wstring path = exe_dir + L"\\dlssnr_on_amd.log";
        if (panel_log.open(path))
            log(0, "NeuralHub's in-game panel (End) is shown over the video when it opens");
        else
            log(1, "NeuralHub's log (%ls) could not be opened: its in-game panel is not shown", path.c_str());
    }
    const int event = panel_log.poll();
    if (event) {
        panel_open = event > 0;
        log(0, "NeuralHub's in-game panel %s", panel_open ? "opened" : "closed");
    }
    HWND owner = panel_owner;
    bool want = panel_open && !panel_failed && owner && IsWindow(owner) && IsWindowVisible(owner) && !IsIconic(owner);
    if (want) {
        HWND fg = GetForegroundWindow();
        want = fg && (fg == owner || fg == hwnd || GetAncestor(fg, GA_ROOTOWNER) == owner);
    }
    if (!want) {
        panelHide();
        return;
    }
    const UINT w = static_cast<UINT>(nr_cfg.out_width), h = static_cast<UINT>(nr_cfg.out_height);
    if (!panelResize(w, h, nr_cfg.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM)) {
        panel_failed = true;
        panelHide();
        return;
    }
    if (nr_cfg.hdr && !panel_desc_ready && !cs_failed) {
        // The frame in scRGB units through a compute pass; without it it is copied as it is
        // (2.5 times darker on an HDR desktop).
        if (!nr_panel.res && !makeTex(nr_panel, static_cast<int>(w), static_cast<int>(h), DXGI_FORMAT_R16G16B16A16_FLOAT, 8))
            nr_panel = Tex();
        if (nr_panel.res && createProxyPipeline()) {
            const D3D12_CPU_DESCRIPTOR_HANDLE base = cs_heap->GetCPUDescriptorHandleForHeapStart();
            for (UINT i = 8; i < 11; i++) {
                D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
                d.Format = nr_out.format;
                d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                d.Texture2D.MipLevels = 1;
                D3D12_CPU_DESCRIPTOR_HANDLE at = base;
                at.ptr += static_cast<SIZE_T>(i) * cs_increment;
                device->CreateShaderResourceView(nr_out.res.Get(), &d, at);
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
            u.Format = nr_panel.format;
            u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            D3D12_CPU_DESCRIPTOR_HANDLE at = base;
            at.ptr += static_cast<SIZE_T>(11) * cs_increment;
            device->CreateUnorderedAccessView(nr_panel.res.Get(), nullptr, &u, at);
            panel_desc_ready = true;
        }
    }
    if (!panel_styled) {
        SetWindowLongPtrW(hwnd, GWL_STYLE, WS_POPUP);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        panel_styled = true;
    }
    // Over the video: the frame's aspect fitted into chiaki's client area.
    RECT client = {};
    POINT origin = {};
    if (!GetClientRect(owner, &client) || !ClientToScreen(owner, &origin) || client.right <= 0 || client.bottom <= 0) {
        panelHide();
        return;
    }
    const double scale = std::min(static_cast<double>(client.right) / w, static_cast<double>(client.bottom) / h);
    const LONG pw = std::max<LONG>(1, static_cast<LONG>(w * scale + 0.5)),
               ph = std::max<LONG>(1, static_cast<LONG>(h * scale + 0.5));
    RECT r;
    r.left = origin.x + (client.right - pw) / 2;
    r.top = origin.y + (client.bottom - ph) / 2;
    r.right = r.left + pw;
    r.bottom = r.top + ph;
    if (panel_shown && !EqualRect(&r, &panel_rect))
        SetWindowPos(hwnd, HWND_TOPMOST, r.left, r.top, pw, ph, SWP_NOACTIVATE);
    panel_rect = r;
    if (nr_cfg.hdr) {
        // Windows' HDR switch can change while streaming, and the window can move to another monitor.
        const HMONITOR monitor = MonitorFromRect(&r, MONITOR_DEFAULTTONEAREST);
        const ULONGLONG now = GetTickCount64();
        if (monitor != panel_monitor || now - panel_monitor_checked > 2000) {
            panel_monitor = monitor;
            panel_monitor_checked = now;
            const int hdr = monitorShowsHdr(monitor) ? 1 : 0;
            if (hdr != panel_monitor_hdr) {
                panel_monitor_hdr = hdr;
                panel_scrgb = hdr ? 203.0f / 80.0f : 1.0f;
                log(0, "NeuralHub's panel window is on a monitor with HDR %s: HDR frames go in scaled by %.2f", hdr ? "on" : "off",
                    panel_scrgb);
            }
        }
    }
    panel_copy = true;
}

// In the evaluation's command list, after the output: the frame into the back buffer.
void Dx12Fg::Impl::panelRecordCopy()
{
    if (!panel_copy)
        return;
    ComPtr<IDXGISwapChain3> sc3;
    ComPtr<ID3D12Resource> back;
    if (FAILED(swapchain.As(&sc3)) || FAILED(sc3->GetBuffer(sc3->GetCurrentBackBufferIndex(), IID_PPV_ARGS(&back)))) {
        panel_copy = false;
        return;
    }
    Tex *src = &nr_out;
    if (nr_cfg.hdr && panel_desc_ready && cs_scrgb) {
        barrier(nr_out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        barrier(nr_panel, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap *heaps[] = { cs_heap.Get() };
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(cs_root.Get());
        list->SetPipelineState(cs_scrgb.Get());
        D3D12_GPU_DESCRIPTOR_HANDLE table = cs_heap->GetGPUDescriptorHandleForHeapStart();
        table.ptr += static_cast<UINT64>(8) * cs_increment;
        list->SetComputeRootDescriptorTable(0, table);
        const UINT w = static_cast<UINT>(nr_cfg.out_width), h = static_cast<UINT>(nr_cfg.out_height);
        const float to_scrgb = panel_scrgb;
        UINT constants[4] = { w, h, 0, 0 };
        std::memcpy(&constants[2], &to_scrgb, sizeof(to_scrgb));
        list->SetComputeRoot32BitConstants(1, 4, constants, 0);
        list->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
        src = &nr_panel;
    }
    barrier(*src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = back.Get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &b);
    list->CopyResource(back.Get(), src->res.Get());
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    list->ResourceBarrier(1, &b);
}

// After the evaluation (and after the signal Vulkan waits for): NeuralHub sees the swapchain
// presented, polls its keys and draws its panel; the window appears once it has a frame.
void Dx12Fg::Impl::panelPresent()
{
    if (!panelRoute() || !mayPresent())
        return;
    const HRESULT hr = swapchain->Present(0, 0);
    if (FAILED(hr) && !panel_present_warned) {
        panel_present_warned = true;
        log(1, "presenting the swapchain NeuralHub's panel is drawn on failed (0x%08x)", static_cast<unsigned>(hr));
    }
    if (panel_copy && !panel_shown) {
        SetWindowPos(hwnd, HWND_TOPMOST, panel_rect.left, panel_rect.top, panel_rect.right - panel_rect.left,
                     panel_rect.bottom - panel_rect.top, SWP_NOACTIVATE | SWP_SHOWWINDOW);
        panel_shown = true;
    }
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
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Depth", nr_depth.res.Get());
        NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.MVec", nr_motion.res.Get());
        NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Enabled", 1);
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
        // A pass whose controls changed starts a new history, and so does every pass after it
        // (its input changed).
        int first_reset = reset ? 0 : nr_cfg.passes;
        for (int pass = 0; pass < first_reset; pass++) {
            if (!nr_tuning_valid || nr_tuning_set[static_cast<size_t>(pass)] != nr_tuning[static_cast<size_t>(pass)])
                first_reset = pass;
        }
#if defined(CHIAKI_DX12FG_TEST_HOOKS)
        if (nrTestKeys())
            first_reset = 0;
#endif
        if (first_reset < nr_cfg.passes && !reset && GetTickCount64() - nr_tuning_logged > 1000) {
            nr_tuning_logged = GetTickCount64();
            const NrTuning &t = nr_tuning[static_cast<size_t>(first_reset)];
            log(0, "DLSS 5 controls (pass %d of %d): style %d, intensity %.2f, local structure %.2f, local tone %.2f, "
                   "auto mask %d, skin structure %.2f",
                first_reset + 1, nr_cfg.passes, t.style, t.intensity, t.local_structure, t.local_tone,
                t.auto_mask ? 1 : 0, t.skin_structure);
        }
        // Passes in a row: model_in -> chain[0] -> chain[1] -> model_out.
        for (int pass = 0; pass < nr_cfg.passes; pass++) {
            Tex &in = pass == 0 ? model_in : nr_chain[static_cast<size_t>((pass - 1) % 2)];
            Tex &out = pass == nr_cfg.passes - 1 ? model_out : nr_chain[static_cast<size_t>(pass % 2)];
            barrier(in, srv);
            barrier(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Color", in.res.Get());
            NVSDK_NGX_Parameter_SetD3d12Resource(nr_params, "DLSSNR.Output", out.res.Get());
            NVSDK_NGX_Parameter_SetUI(nr_params, "DLSSNR.Reset", pass >= first_reset ? 1 : 0);
            nrSetKeys(pass);
            stage = "nvngx_dlssnr EvaluateFeature";
            NVSDK_NGX_Handle *handle = pass == 0 ? nr_handle : nr_more[static_cast<size_t>(pass - 1)];
            const auto r = static_cast<NVSDK_NGX_Result>(gate.evaluate(list.Get(), handle, nr_params));
            if (NVSDK_NGX_FAILED(r)) {
                log(log_level, "evaluating DLSS 5 neural rendering (feature 18, pass %d) failed: %s (0x%08x) %s",
                    pass + 1, ngxResultString(r), static_cast<unsigned>(r), gate.fault());
                return false;
            }
        }
        nr_tuning_set = nr_tuning;
        nr_tuning_valid = true;
        NVSDK_NGX_Result r = NVSDK_NGX_Result_Success;
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
    for (auto &h : nr_more) {
        if (h)
            gate.release(h);
        h = nullptr;
    }
    nr_tuning_valid = false;
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
    for (auto &t : nr_chain)
        t = Tex();
    nr_cfg = NrConfig();
    nr_sized = false;
    nr_query_done = 0;
    panelHide();
    nr_panel = Tex();
    panel_desc_ready = false;
    panel_failed = false;
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
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (d.sl_hwnd) {
        {
            std::lock_guard<std::mutex> lock(g_sl_capture_mutex);
            if (g_sl_capture_owner == &d)
                g_sl_capture_owner = nullptr;
        }
        if (g_sl_hwnd.load() == d.sl_hwnd)
            g_sl_hwnd.store(nullptr);
        DestroyWindow(d.sl_hwnd);
    }
    if (d.sl_readback)
        d.sl_readback->Unmap(0, nullptr);
    if (d.sl_cap_event)
        CloseHandle(d.sl_cap_event);
#endif
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
    if (d.ngx_up)
        g_ngx_d3d12_up.fetch_sub(1);
    d.ngx_up = false;
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
    if (!d.createSwapchain() || !d.initNgx() || !d.createSharedFence())
        return false;
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (d.sl_route) {
        if (d.slSetup()) {
            d.fg_available = true;
            d.max_generated = d.sl_max_generated;
        } else {
            d.log(1, "Streamline could not be set up; DLSS Frame Generation goes through NGX");
            d.sl_route = false;
        }
    }
#endif
    return true;
}

int Dx12Fg::maxGenerated() const
{
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (impl->sl_route)
        return std::max(1, std::min(impl->max_generated, impl->sl_cap));
#endif
    return impl->max_generated;
}

void Dx12Fg::setFrameInterval(float ms)
{
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (std::isfinite(ms) && ms > 1.0f && ms < 200.0f)
        impl->sl_stream_interval = ms;
#else
    (void)ms;
#endif
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
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (d.sl_route) {
        if (!d.slEnsureSize(w, h)) {
            d.log(1, "Streamline: frame generation could not be set up for %dx%d", w, h);
            d.destroySized();
            return false;
        }
        d.sized = true;
        d.log(0, "DLSS Frame Generation (Streamline) ready: %dx%d", w, h);
        return true;
    }
#endif
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

uint64_t Dx12Fg::completedValue() const
{
    // No fence: nothing can be waited for, report everything as done.
    return impl->shared_fence ? impl->shared_fence->GetCompletedValue() : UINT64_MAX;
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
    if (!d.motion_buf.buffer)
        return; // Streamline: DLSS-G finds the motion itself
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
    const Impl::Tex *out = &d.outs[static_cast<size_t>(index)];
    VkBuffer buffer = d.out_bufs[static_cast<size_t>(index)].buffer;
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (d.sl_route) {
        // The index-th generated frame takeFrames() handed out, in its frame slot.
        out = &d.color;
        buffer = d.sl_ring[static_cast<size_t>(d.sl_taken[static_cast<size_t>(index)])].buf.buffer;
    }
#endif
    VkBufferImageCopy region = {};
    region.bufferRowLength = out->footprint.Footprint.RowPitch / out->bytes_per_texel;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(d.width), static_cast<uint32_t>(d.height), 1 };
    d.vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

bool Dx12Fg::asyncFrames() const
{
    return impl->sl_route;
}

bool Dx12Fg::framesReady()
{
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (impl->sl_route)
        return impl->slFramesReady();
#endif
    return true;
}

int Dx12Fg::takeFrames()
{
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (impl->sl_route)
        return impl->slTakeFrames();
#endif
    return 0;
}

bool Dx12Fg::generate(int count, uint64_t wait_value, uint64_t signal_value, bool reset, bool use_motion,
                      float mvec_scale_x, float mvec_scale_y)
{
    Impl &d = *impl;
#if defined(CHIAKI_HAVE_STREAMLINE)
    if (d.sl_route)
        return d.slGenerate(count, wait_value, signal_value, reset);
#endif
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
        || config.out_height < config.height || config.passes < 1 || config.passes > kMaxNrPasses) {
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
    // Between feature 18 passes: what the model reads and writes (RGBA8; HDR goes through the
    // RGBA8 proxy).
    for (int i = 0; ok && d.gate_ready && i < std::min(config.passes - 1, 2); i++)
        ok = d.makeTex(d.nr_chain[static_cast<size_t>(i)], w, h, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
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
    // NeuralHub's panel window: the swapchain takes the output's size and format now, while
    // the panel is closed, so opening it needs no resize.
    if (!d.gate_ready && d.swapchain
        && !d.panelResize(static_cast<UINT>(ow), static_cast<UINT>(oh), color))
        d.panel_failed = true;
    std::string route;
    if (d.gate_ready) {
        route = std::string("NGX feature 18 ") + (config.model == 1 ? "HT" : "CC");
        if (config.passes > 1)
            route += " x" + std::to_string(config.passes) + " passes";
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

void Dx12Fg::nrSetTuning(int pass, const NrTuning &tuning)
{
    if (pass < 0 || pass >= kMaxNrPasses)
        return;
    NrTuning t = tuning;
    t.style = std::clamp(t.style, 0, 2);
    t.intensity = std::isfinite(t.intensity) ? std::clamp(t.intensity, 0.0f, 1.0f) : 1.0f;
    t.local_structure = std::isfinite(t.local_structure) ? std::clamp(t.local_structure, 0.0f, 2.0f) : 1.0f;
    t.local_tone = std::isfinite(t.local_tone) ? std::clamp(t.local_tone, 0.0f, 2.0f) : 1.0f;
    t.skin_structure = std::isfinite(t.skin_structure) ? std::clamp(t.skin_structure, 0.0f, 2.5f) : 1.0f;
    impl->nr_tuning[static_cast<size_t>(pass)] = t;
}

bool Dx12Fg::nrTunable() const
{
    return impl->gate_ready;
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
    // Before the queue waits for Vulkan: it may resize the swapchain (waits for the queue).
    d.stage = "NeuralHub's panel";
    d.panelUpdate();
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
            if (ok) {
                d.copyTexToBuffer(d.nr_out_buf.res.Get(), d.nr_out);
                d.panelRecordCopy();
            }
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
    const bool signalled = d.signalShared(signal_value);
    if (ok) {
        d.stage = "Present (NeuralHub's panel)";
        d.panelPresent();
    } else {
        d.panel_copy = false;
    }
    return signalled && ok;
}

void Dx12Fg::nrSetPanelWindow(void *window)
{
    impl->panel_owner = static_cast<HWND>(window);
}

bool Dx12Fg::nrProviderPanel() const
{
    return !impl->gate_ready && nrAvailable();
}

void Dx12Fg::nrHidePanel()
{
    impl->panelHide();
}

void Dx12Fg::markCrashed()
{
    impl->crashed = true;
}

bool Dx12Fg::crashed() const
{
    return impl->crashed;
}

bool Dx12Fg::ngxUpInProcess()
{
    return g_ngx_d3d12_up.load() > 0;
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
    if (created->impl->sl_route) {
        // Streamline is given one device for the process: this instance stays (never destroyed,
        // also not at exit, after the Vulkan device is gone).
        static auto *kept = new std::shared_ptr<Dx12Fg>();
        *kept = created;
    }
    return created;
}

const char *Dx12Fg::stage() const
{
    return impl->stage;
}
