// nrgate: built as "nvngx.dll_dlssnr.dll". NVIDIA's DLSS Neural Rendering snippet
// (nvngx_dlssnr.dll) looks up the module its caller returns into and only serves modules whose
// file name contains "nvngx.dll". This DLL is a thin pass-through with such a name: the host
// fills every NGX parameter itself and calls the snippet through here. It holds no NVIDIA code.
// Every call is wrapped in SEH so a fault inside the snippet comes back as an error code and a
// readable note instead of ending the process.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>

namespace {

using InitFn = int(__cdecl *)(unsigned long long, const wchar_t *, void *, int, const void *);
using CreateFn = int(__cdecl *)(void *, int, void *, void **);
using EvalFn = int(__cdecl *)(void *, void *, void *, void *);
using ReleaseFn = int(__cdecl *)(void *);
using ShutdownFn = int(__cdecl *)(void *);

HMODULE g_module = nullptr;
InitFn g_init = nullptr;
CreateFn g_create = nullptr;
EvalFn g_eval = nullptr;
ReleaseFn g_release = nullptr;
ShutdownFn g_shutdown = nullptr;
char g_fault[640] = {};

// Not NVSDK_NGX_Result_Fail itself, but still inside the 0xBAD00000 failure range.
constexpr int kFaulted = static_cast<int>(0xBAD000FF);

int noteFault(EXCEPTION_POINTERS *ep, const char *stage)
{
    unsigned code = 0;
    void *addr = nullptr;
    if (ep && ep->ExceptionRecord) {
        code = ep->ExceptionRecord->ExceptionCode;
        addr = ep->ExceptionRecord->ExceptionAddress;
    }
    char mod[MAX_PATH] = "?";
    unsigned long long offset = 0;
    HMODULE m = nullptr;
    if (addr && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCSTR>(addr), &m)
        && m) {
        GetModuleFileNameA(m, mod, MAX_PATH);
        offset = reinterpret_cast<unsigned long long>(addr) - reinterpret_cast<unsigned long long>(m);
    }
    snprintf(g_fault, sizeof(g_fault), "%s faulted: exception 0x%08X at %s+0x%llX", stage, code, mod, offset);
    return EXCEPTION_EXECUTE_HANDLER;
}

// No objects with destructors in these (__try needs that), and the result goes through a
// volatile so the snippet is called, never tail-jumped to: its caller must be this module.
int guardedInit(unsigned long long app, const wchar_t *data, void *device, int sdk, const void *params)
{
    __try {
        volatile int r = g_init(app, data, device, sdk, params);
        return r;
    } __except (noteFault(GetExceptionInformation(), "Init_Ext")) {
        return kFaulted;
    }
}

int guardedCreate(void *cmd, int feature, void *params, void **handle)
{
    __try {
        volatile int r = g_create(cmd, feature, params, handle);
        return r;
    } __except (noteFault(GetExceptionInformation(), "CreateFeature")) {
        return kFaulted;
    }
}

int guardedEvaluate(void *cmd, void *handle, void *params)
{
    __try {
        volatile int r = g_eval(cmd, handle, params, nullptr);
        return r;
    } __except (noteFault(GetExceptionInformation(), "EvaluateFeature")) {
        return kFaulted;
    }
}

int guardedRelease(void *handle)
{
    __try {
        volatile int r = g_release(handle);
        return r;
    } __except (noteFault(GetExceptionInformation(), "ReleaseFeature")) {
        return kFaulted;
    }
}

int guardedShutdown(void *device)
{
    __try {
        volatile int r = g_shutdown(device);
        return r;
    } __except (noteFault(GetExceptionInformation(), "Shutdown1")) {
        return kFaulted;
    }
}

} // namespace

extern "C" {

// Loads the snippet from an absolute path. 1 when every entry point was found.
__declspec(dllexport) int nrgate_load(const wchar_t *snippet_path)
{
    if (!g_module) {
        g_module = LoadLibraryExW(snippet_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g_module) {
            snprintf(g_fault, sizeof(g_fault), "LoadLibrary failed, error %lu", GetLastError());
            return 0;
        }
        g_init = reinterpret_cast<InitFn>(GetProcAddress(g_module, "NVSDK_NGX_D3D12_Init_Ext"));
        g_create = reinterpret_cast<CreateFn>(GetProcAddress(g_module, "NVSDK_NGX_D3D12_CreateFeature"));
        g_eval = reinterpret_cast<EvalFn>(GetProcAddress(g_module, "NVSDK_NGX_D3D12_EvaluateFeature"));
        g_release = reinterpret_cast<ReleaseFn>(GetProcAddress(g_module, "NVSDK_NGX_D3D12_ReleaseFeature"));
        g_shutdown = reinterpret_cast<ShutdownFn>(GetProcAddress(g_module, "NVSDK_NGX_D3D12_Shutdown1"));
    }
    if (!(g_init && g_create && g_eval && g_release)) {
        snprintf(g_fault, sizeof(g_fault), "snippet lacks an NGX D3D12 entry point");
        return 0;
    }
    return 1;
}

__declspec(dllexport) int nrgate_init(unsigned long long app_id, const wchar_t *data_path, void *device, int sdk_version,
                                      const void *params)
{
    g_fault[0] = 0;
    return g_init ? guardedInit(app_id, data_path, device, sdk_version, params) : 0;
}

__declspec(dllexport) int nrgate_create(void *cmd, int feature, void *params, void **handle)
{
    g_fault[0] = 0;
    return g_create ? guardedCreate(cmd, feature, params, handle) : 0;
}

__declspec(dllexport) int nrgate_evaluate(void *cmd, void *handle, void *params)
{
    g_fault[0] = 0;
    return g_eval ? guardedEvaluate(cmd, handle, params) : 0;
}

__declspec(dllexport) int nrgate_release(void *handle)
{
    g_fault[0] = 0;
    return (g_release && handle) ? guardedRelease(handle) : 0;
}

// Drops what Init_Ext set up for this device (before the device goes away).
__declspec(dllexport) int nrgate_shutdown(void *device)
{
    g_fault[0] = 0;
    return (g_shutdown && device) ? guardedShutdown(device) : 0;
}

// Empty when the last call raised nothing.
__declspec(dllexport) const char *nrgate_fault() { return g_fault; }

} // extern "C"

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
