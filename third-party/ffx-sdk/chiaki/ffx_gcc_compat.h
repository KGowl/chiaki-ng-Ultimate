// chiaki-ng: force-included when the FidelityFX SDK sources are built with
// GCC/Clang on Linux. The SDK is written for MSVC, which provides these
// implicitly.
#pragma once
#include <math.h>
#include <string.h>
#include <wchar.h>
#ifdef __cplusplus
#include <algorithm>
#include <codecvt>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <locale>
#include <mutex>
#include <string>
#endif

#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#endif
#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))
#endif

#ifdef __cplusplus
// MSVC bounds-checked string functions used by the SDK.
inline int wcscpy_s(wchar_t *dst, size_t size, const wchar_t *src)
{
    if (!dst || !size)
        return 22;
    size_t i = 0;
    for (; src && src[i] && i + 1 < size; ++i)
        dst[i] = src[i];
    dst[i] = 0;
    return 0;
}
template <size_t N> inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t *src) { return wcscpy_s(dst, N, src); }
inline int strcpy_s(char *dst, size_t size, const char *src)
{
    if (!dst || !size)
        return 22;
    std::snprintf(dst, size, "%s", src ? src : "");
    return 0;
}
template <size_t N> inline int strcpy_s(char (&dst)[N], const char *src) { return strcpy_s(dst, N, src); }
inline int sprintf_s(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    const int r = std::vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    return r;
}
template <size_t N> inline int sprintf_s(char (&buf)[N], const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    const int r = std::vsnprintf(buf, N, fmt, ap);
    va_end(ap);
    return r;
}
#endif
