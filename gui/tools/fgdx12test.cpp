// chiaki-fgdx12test: console test of chiaki-ng's frame generation backends, driven the way
// the renderer drives them: synthetic frames of a moving pattern go through FrameGen::process
// on a libplacebo Vulkan device, and the generated frames are read back and compared with
// where the pattern should be between the previous and the current real frame.
//
// Usage: chiaki-fgdx12test [--backend 5] [--frames N] [--mvx N] [--w W --h H] [--hdr]
//   --backend  FrameGen::Backend value: 1 DLSS (Vulkan), 3 Flow, 5 DLSS through D3D12 (default)
//   --hdr      HDR frame generation: the pattern as a PQ BT.2020 RGBA16F frame (up to 1500 nits),
//              checked in PQ terms: the generated frames sit where they should, keep the highlights
//              (pixels above SDR white, 203 nits, as in the source) and keep more than 8 bits.
//        chiaki-fgdx12test --nr [--image file] [--out file.png] [--with-fg] [--frames N] [--mvx N]
//                          [--scale 1|1.5|2] [--hdr] [--model 0|1] [--passes 1-3] [--tune-at K]
//   --nr       neural rendering (NeuralRender, DLSS 5 through D3D12) on a moving pattern or on
//              an image panned by --mvx px per frame; --out saves the last output, --with-fg
//              also runs DLSS Frame Generation through D3D12 on the output (shared runtime),
//              --scale upscales after it, --hdr feeds a linear-light HDR frame, --model picks
//              NVIDIA's feature 18 model (0 CC, 1 HT).
//   --pipeline drives backend 5 and --nr (with --with-fg) the way the renderer pipelines them:
//              begin(), poll beginDone(), then process() / finish(), with the same checks.
//   --timing   (frame generation) times the calls instead of checking the frames: one frame every
//              16.7 ms like a 60 fps stream, nothing read back, median / p90 / max per phase. The
//              checks read every frame back on the CPU and leave the GPU idle long enough to clock
//              down, so their times are not what a stream gets.
//        chiaki-fgdx12test --eval <dir> [--backend 5] [--multiplier 2-4] [--cleanup 0-2] [--dlss-motion 0-2]
//                          [--flow-hq] [--sdr] [--fps 60] [--start N] [--count N] [--out <dir>] [--save-worst N]
//   --eval     frame generation on a real 60 fps capture (PNG files in name order; 16-bit = HDR, PQ BT.2020
//              as decoded): every multiplier-th frame is fed as a real frame and each generated frame is
//              compared with the real frame it stands for, next to showing the last real frame again and
//              to a plain blend. Per frame: luma PSNR / MAE, "art" = 16x16 blocks off by more than 12/255,
//              "worse" = blocks clearly worse than repeating the last real frame. --out keeps the worst
//              frames as truth | generated | difference pictures. Scene cuts reset it as in the renderer.
// For backend 5 and --nr on a GPU without NVIDIA's runtime, NeuralHub's dxgi.dll and
// neuralhub.ini must sit next to the executable.

#include "framegen.h"
#include "neuralrender.h"

#include <QImage>

#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>
#include <nvsdk_ngx_vk.h>

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#endif

Q_LOGGING_CATEGORY(chiakiGui, "chiaki.gui", QtInfoMsg)

namespace {

// --pipeline: waits (up to 5 s) for a begun call's D3D12 side, returns the wait in ms or -1.
template <typename Done> double waitBegun(Done done)
{
    QElapsedTimer wait;
    wait.start();
    while (!done()) {
        if (wait.elapsed() > 5000)
            return -1.0;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return wait.nsecsElapsed() / 1e6;
}

void placeboLog(void *, pl_log_level level, const char *msg)
{
    if (level <= PL_LOG_WARN)
        std::fprintf(stderr, "[placebo] %s\n", msg);
}

uint8_t pattern(int x, int y, int shift_px)
{
    const int xx = x - shift_px + 1000000;
    const int v = 40 + (xx * 3) % 97 + ((((xx >> 5) ^ (y >> 5)) & 1) * 80) + (y * 5) % 23;
    return static_cast<uint8_t>(std::min(v, 255));
}

float srgbToLinear(float v)
{
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

uint16_t floatToHalf(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// ---- HDR frame generation test (--hdr without --nr) ----
// The pattern's values (0..255) as light: up to kPqPeak nits, about half of the pixels above SDR
// white, PQ encoded (BT.2020) into a half float frame.
constexpr double kPqPeak = 1500.0;

double pqEncode(double nits)
{
    const double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0;
    const double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
    const double y = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), m1);
    return std::pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

double pqDecode(double e)
{
    const double m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0;
    const double c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
    const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / m2);
    return 10000.0 * std::pow(std::max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

double patternNits(double v)
{
    return 0.5 + kPqPeak * std::pow(std::clamp(v, 0.0, 255.0) / 255.0, 2.4);
}

// Back from light to the pattern's scale, so bestShift's thresholds stay what they are in SDR.
float patternFromNits(double nits)
{
    return static_cast<float>(255.0 * std::pow(std::max(nits - 0.5, 0.0) / kPqPeak, 1.0 / 2.4));
}

struct PqStats
{
    double peak_nits = 0.0; // brightest pixel
    double above_sdr = 0.0; // share of pixels above SDR white (scRGB > 1.0), percent
    double grid = 0.0;      // mean distance of PQ * 255 from the 8-bit grid: ~0.25 above 8 bits, ~0 at 8 bits
};

PqStats pqStats(const std::vector<float> &pq)
{
    PqStats s;
    const size_t n = pq.size() / 4;
    for (size_t i = 0; i < pq.size(); i += 4) {
        const double nits = pqDecode(pq[i]);
        s.peak_nits = std::max(s.peak_nits, nits);
        if (nits > PL_COLOR_SDR_WHITE)
            s.above_sdr += 1.0;
        const double q = pq[i] * 255.0;
        s.grid += std::fabs(q - std::round(q));
    }
    s.above_sdr = 100.0 * s.above_sdr / std::max<size_t>(n, 1);
    s.grid /= std::max<size_t>(n, 1);
    return s;
}

struct Rig
{
    pl_log log = nullptr;
    pl_vk_inst inst = nullptr;
    pl_vulkan vk = nullptr;
    pl_gpu gpu = nullptr;
    pl_renderer renderer = nullptr;
    pl_tex source = nullptr;
    pl_tex readback = nullptr;
    int w = 0;
    int h = 0;

    bool init()
    {
        struct pl_log_params log_params = {};
        log_params.log_cb = placeboLog;
        log_params.log_level = PL_LOG_WARN;
        log = pl_log_create(PL_API_VER, &log_params);
        // NGX's Vulkan extensions, optional as in chiaki-ng (RtxVsr::appendRequiredExtensions):
        // without them DLSS Frame Generation through Vulkan (--backend 1) fails with PlatformError.
        unsigned int ngx_inst_count = 0, ngx_dev_count = 0;
        const char **ngx_inst = nullptr, **ngx_dev = nullptr;
        if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ngx_inst_count, &ngx_inst, &ngx_dev_count, &ngx_dev))) {
            ngx_inst_count = ngx_dev_count = 0;
            ngx_inst = ngx_dev = nullptr;
        }
        struct pl_vk_inst_params inst_params = {};
        inst_params.opt_extensions = ngx_inst;
        inst_params.num_opt_extensions = static_cast<int>(ngx_inst_count);
        inst = pl_vk_inst_create(log, &inst_params);
        if (!inst)
            return false;
        struct pl_vulkan_params vulkan_params = {
            .instance = inst->instance,
            .get_proc_addr = inst->get_proc_addr,
            .allow_software = false,
            PL_VULKAN_DEFAULTS
            .opt_extensions = ngx_dev,
            .num_opt_extensions = static_cast<int>(ngx_dev_count),
        };
        vk = pl_vulkan_create(log, &vulkan_params);
        if (!vk)
            return false;
        gpu = vk->gpu;
        renderer = pl_renderer_create(log, gpu);
        return renderer != nullptr;
    }

    bool resize(int new_w, int new_h)
    {
        if (source)
            pl_tex_destroy(gpu, &source);
        if (readback)
            pl_tex_destroy(gpu, &readback);
        w = new_w;
        h = new_h;
        pl_fmt fmt = pl_find_named_fmt(gpu, "rgba8");
        if (!fmt)
            return false;
        struct pl_tex_params src_params = {};
        src_params.w = w;
        src_params.h = h;
        src_params.format = fmt;
        src_params.sampleable = true;
        src_params.host_writable = true;
        source = pl_tex_create(gpu, &src_params);
        struct pl_tex_params rb_params = {};
        rb_params.w = w;
        rb_params.h = h;
        rb_params.format = fmt;
        rb_params.renderable = true;
        rb_params.host_readable = true;
        readback = pl_tex_create(gpu, &rb_params);
        return source && readback;
    }

    void upload(int shift_px)
    {
        std::vector<uint8_t> data(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; y++) {
            uint8_t *row = data.data() + static_cast<size_t>(y) * w * 4;
            for (int x = 0; x < w; x++) {
                const uint8_t v = pattern(x, y, shift_px);
                row[4 * x] = row[4 * x + 1] = row[4 * x + 2] = v;
                row[4 * x + 3] = 255;
            }
        }
        uploadRgba(data.data());
    }

    void uploadRgba(const uint8_t *data)
    {
        struct pl_tex_transfer_params up = {};
        up.tex = source;
        up.ptr = const_cast<uint8_t *>(data);
        up.row_pitch = static_cast<size_t>(w) * 4;
        pl_tex_upload(gpu, &up);
    }

    struct pl_frame sourceFrame() const
    {
        struct pl_frame frame = {};
        frame.num_planes = 1;
        frame.planes[0].texture = source;
        frame.planes[0].components = 4;
        for (int i = 0; i < 4; i++)
            frame.planes[0].component_mapping[i] = i;
        frame.repr = pl_color_repr_rgb;
        frame.color = pl_color_space_bt709;
        frame.crop = { 0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h) };
        return frame;
    }

    // Reads a frame produced by FrameGen back to host memory (red channel, one byte per pixel).
    bool read(const struct pl_frame &frame, std::vector<uint8_t> &red)
    {
        std::vector<uint8_t> rgba;
        if (!readRgba(frame, rgba))
            return false;
        red.resize(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < red.size(); i++)
            red[i] = rgba[4 * i];
        return true;
    }

    bool readRgba(const struct pl_frame &frame, std::vector<uint8_t> &rgba)
    {
        struct pl_frame target = {};
        target.num_planes = 1;
        target.planes[0].texture = readback;
        target.planes[0].components = 4;
        for (int i = 0; i < 4; i++)
            target.planes[0].component_mapping[i] = i;
        target.repr = pl_color_repr_rgb;
        target.color = pl_color_space_bt709;
        struct pl_render_params params = pl_render_default_params;
        params.frame_mixer = nullptr;
        params.dither_params = nullptr;
        if (!pl_render_image(renderer, &frame, &target, &params))
            return false;
        rgba.resize(static_cast<size_t>(w) * h * 4);
        struct pl_tex_transfer_params dn = {};
        dn.tex = readback;
        dn.ptr = rgba.data();
        dn.row_pitch = static_cast<size_t>(w) * 4;
        return pl_tex_download(gpu, &dn);
    }

    // ---- neural rendering test: HDR source and readback at any size ----
    pl_tex source_hdr = nullptr; // rgba16hf, linear light
    pl_tex float_target = nullptr;
    float hdr_peak = 8.0f;

    bool ensureHdrSource()
    {
        if (source_hdr && source_hdr->params.w == w && source_hdr->params.h == h)
            return true;
        if (source_hdr)
            pl_tex_destroy(gpu, &source_hdr);
        struct pl_tex_params p = {};
        p.w = w;
        p.h = h;
        // "rgba16hf" takes half floats from the host; "rgba16f" is the same texture with a float32 upload format.
        p.format = pl_find_named_fmt(gpu, "rgba16hf");
        p.sampleable = true;
        p.host_writable = true;
        source_hdr = p.format ? pl_tex_create(gpu, &p) : nullptr;
        return source_hdr != nullptr;
    }

    void uploadHalf(const uint16_t *data)
    {
        struct pl_tex_transfer_params up = {};
        up.tex = source_hdr;
        up.ptr = const_cast<uint16_t *>(data);
        up.row_pitch = static_cast<size_t>(w) * 8;
        pl_tex_upload(gpu, &up);
    }

    // --hdr frame generation: the pattern PQ encoded (BT.2020) into the half float source.
    struct pl_color_space pqSpace() const
    {
        struct pl_color_space c = {};
        c.primaries = PL_COLOR_PRIM_BT_2020;
        c.transfer = PL_COLOR_TRC_PQ;
        c.hdr.max_luma = static_cast<float>(kPqPeak);
        c.hdr.min_luma = 0.005f;
        c.hdr.prim = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020);
        return c;
    }

    bool uploadPq(int shift_px)
    {
        if (!ensureHdrSource())
            return false;
        // The pattern has 256 levels: their PQ halves once, not a pow() per pixel per frame. Computing
        // them per pixel kept a CPU core busy long enough between frames for the GPU to clock down,
        // which made HDR frame generation look 4-7x slower than SDR here (it is not in a stream).
        static const std::array<uint16_t, 256> pq_half = [] {
            std::array<uint16_t, 256> lut = {};
            for (int i = 0; i < 256; i++)
                lut[static_cast<size_t>(i)] = floatToHalf(static_cast<float>(pqEncode(patternNits(i))));
            return lut;
        }();
        std::vector<uint16_t> data(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; y++) {
            uint16_t *row = data.data() + static_cast<size_t>(y) * w * 4;
            for (int x = 0; x < w; x++) {
                const uint16_t v = pq_half[pattern(x, y, shift_px)];
                row[4 * x] = row[4 * x + 1] = row[4 * x + 2] = v;
                row[4 * x + 3] = 0x3c00; // 1.0
            }
        }
        uploadHalf(data.data());
        return true;
    }

    struct pl_frame pqSourceFrame() const
    {
        struct pl_frame frame = sourceFrame();
        frame.planes[0].texture = source_hdr;
        frame.color = pqSpace();
        return frame;
    }

    // A frame produced by FrameGen in PQ terms (rgba32f, no tone mapping: it is in that space)
    // and its red channel on the pattern's scale.
    bool readPq(const struct pl_frame &frame, std::vector<float> &pq, std::vector<float> &red)
    {
        if (!readFloat(frame, w, h, pqSpace(), pq))
            return false;
        red.resize(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < red.size(); i++)
            red[i] = patternFromNits(pqDecode(pq[4 * i]));
        return true;
    }

    // Linear light, 1.0 = SDR white, BT.709, highlights up to hdr_peak: how chiaki-ng's
    // neural rendering takes an HDR stream.
    struct pl_color_space hdrSpace() const
    {
        struct pl_color_space c = {};
        c.primaries = PL_COLOR_PRIM_BT_709;
        c.transfer = PL_COLOR_TRC_LINEAR;
        c.hdr.max_luma = hdr_peak * PL_COLOR_SDR_WHITE;
        c.hdr.prim = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_709);
        return c;
    }

    struct pl_frame hdrSourceFrame() const
    {
        struct pl_frame frame = sourceFrame();
        frame.planes[0].texture = source_hdr;
        frame.color = hdrSpace();
        return frame;
    }

    // Renders `frame` into an ow x oh rgba32f target in `color` and reads it back.
    bool readFloat(const struct pl_frame &frame, int ow, int oh, const struct pl_color_space &color,
                   std::vector<float> &out)
    {
        if (!float_target || float_target->params.w != ow || float_target->params.h != oh) {
            if (float_target)
                pl_tex_destroy(gpu, &float_target);
            struct pl_tex_params p = {};
            p.w = ow;
            p.h = oh;
            p.format = pl_find_named_fmt(gpu, "rgba32f");
            p.renderable = true;
            p.host_readable = true;
            float_target = p.format ? pl_tex_create(gpu, &p) : nullptr;
            if (!float_target)
                return false;
        }
        struct pl_frame target = {};
        target.num_planes = 1;
        target.planes[0].texture = float_target;
        target.planes[0].components = 4;
        for (int i = 0; i < 4; i++)
            target.planes[0].component_mapping[i] = i;
        target.repr = pl_color_repr_rgb;
        target.color = color;
        struct pl_render_params params = pl_render_default_params;
        params.frame_mixer = nullptr;
        params.dither_params = nullptr;
        params.peak_detect_params = nullptr;
        if (!pl_render_image(renderer, &frame, &target, &params))
            return false;
        out.resize(static_cast<size_t>(ow) * oh * 4);
        struct pl_tex_transfer_params dn = {};
        dn.tex = float_target;
        dn.ptr = out.data();
        dn.row_pitch = static_cast<size_t>(ow) * 16;
        return pl_tex_download(gpu, &dn);
    }

    void destroy()
    {
        if (gpu) {
            if (source)
                pl_tex_destroy(gpu, &source);
            if (readback)
                pl_tex_destroy(gpu, &readback);
            if (source_hdr)
                pl_tex_destroy(gpu, &source_hdr);
            if (float_target)
                pl_tex_destroy(gpu, &float_target);
        }
        if (renderer)
            pl_renderer_destroy(&renderer);
        if (vk)
            pl_vulkan_destroy(&vk);
        if (inst)
            pl_vk_inst_destroy(&inst);
        if (log)
            pl_log_destroy(&log);
    }
};

// Best horizontal shift of `red` against the pattern of the current frame (the one at
// `shift_px`): 0 = the current frame, `mvx` = the previous frame. Returns the shift and the
// mean absolute difference there and at 0.
template <typename T>
int bestShift(const std::vector<T> &red, int w, int h, int shift_px, double &diff_best, double &diff_zero)
{
    double dsum[17] = {};
    int count = 0;
    for (int y = h * 4 / 10; y < h * 6 / 10; y += 4) {
        const T *row = red.data() + static_cast<size_t>(y) * w;
        for (int x = 100; x < w - 100; x += 2) {
            for (int s = -8; s <= 8; s++)
                dsum[s + 8] += std::fabs(static_cast<double>(row[x]) - static_cast<double>(pattern(x + s, y, shift_px)));
            count++;
        }
    }
    int best = 0;
    for (int s = 1; s < 17; s++) {
        if (dsum[s] < dsum[best])
            best = s;
    }
    diff_best = dsum[best] / std::max(count, 1);
    diff_zero = dsum[8] / std::max(count, 1);
    return best - 8;
}

struct Phase
{
    const char *name;
    int w, h, multiplier, frames, cleanup;
};

// A synthetic HDR frame from an SDR one (same as nrprobe): linear light with the highlights
// pushed up to `peak` times SDR white.
std::vector<uint16_t> toHdrHalf(const std::vector<uint8_t> &rgba, float peak)
{
    std::vector<uint16_t> out(rgba.size());
    for (size_t i = 0; i < rgba.size(); i += 4) {
        float l[3];
        for (int c = 0; c < 3; c++)
            l[c] = srgbToLinear(rgba[i + c] / 255.0f);
        const float m = std::max({ l[0], l[1], l[2] });
        const float t = std::clamp((m - 0.5f) / 0.5f, 0.0f, 1.0f);
        const float boost = 1.0f + (peak - 1.0f) * t * t * (3.0f - 2.0f * t);
        for (int c = 0; c < 3; c++)
            out[i + c] = floatToHalf(l[c] * boost);
        out[i + 3] = floatToHalf(1.0f);
    }
    return out;
}

struct HdrStats
{
    double mean = 0.0, max = 0.0, above = 0.0; // luminance, share of pixels above SDR white
};

HdrStats hdrStats(const std::vector<float> &f)
{
    HdrStats s;
    const size_t n = f.size() / 4;
    for (size_t i = 0; i < f.size(); i += 4) {
        const double y = 0.2126 * f[i] + 0.7152 * f[i + 1] + 0.0722 * f[i + 2];
        s.mean += y;
        s.max = std::max(s.max, y);
        if (y > 1.0)
            s.above += 1.0;
    }
    s.mean /= std::max<size_t>(n, 1);
    s.above = 100.0 * s.above / std::max<size_t>(n, 1);
    return s;
}

// Neural rendering on a moving pattern or a panned image: every output must be a real
// picture close to its input (not black, not the previous frame), and the runtime must keep
// working when frame generation through D3D12 shares it. --scale upscales after it, --hdr
// feeds a linear-light HDR frame (highlights to 8x SDR white) and checks they stay HDR.
int runNeuralRender(Rig &rig, const std::string &image_path, const std::string &out_path, int frames, int mvx, int w,
                    int h, bool with_fg, float scale, bool hdr, int model, int passes, int tune_at, bool pipeline)
{
    std::vector<uint8_t> base;
    if (!image_path.empty()) {
        QImage img(QString::fromStdString(image_path));
        if (img.isNull()) {
            std::printf("VERDICT: could not read %s\n", image_path.c_str());
            return 4;
        }
        img = img.convertToFormat(QImage::Format_RGBA8888);
        w = img.width();
        h = img.height();
        base.resize(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; y++)
            std::memcpy(base.data() + static_cast<size_t>(y) * w * 4, img.constScanLine(y), static_cast<size_t>(w) * 4);
    }
    NeuralRender::Options options;
    options.model = model;
    options.passes = passes;
    if (scale > 1.01f) {
        options.out_width = static_cast<int>(std::lround(w * scale / 2.0f)) * 2;
        options.out_height = static_cast<int>(std::lround(h * scale / 2.0f)) * 2;
    }
    std::printf("neural rendering test: %dx%d -> %dx%d %s %s, model %s, %d frames, %d px/frame pan%s\n", w, h,
                options.out_width ? options.out_width : w, options.out_height ? options.out_height : h,
                hdr ? "HDR" : "SDR", base.empty() ? "pattern" : image_path.c_str(), model == 1 ? "HT" : "CC", frames,
                mvx, with_fg ? ", with DLSS FG D3D12" : "");
    if (pipeline)
        std::printf("pipelined: begin(), beginDone(), then process()%s\n", with_fg ? " / FG begin(), finish()" : "");
    if (passes > 1 || tune_at >= 0)
        std::printf("%d pass(es)%s\n", passes, tune_at >= 0 ? ", controls changed live (Cinematic, local tone 2) from frame --tune-at on" : "");
    if (!rig.resize(w, h) || (hdr && !rig.ensureHdrSource())) {
        std::printf("VERDICT: textures\n");
        return 3;
    }
    NeuralRender nr(rig.log);
    QElapsedTimer timer;
    timer.start();
    if (!nr.ensureInitialized(rig.vk)) {
        std::printf("VERDICT: neural rendering unavailable (see the log above)\n");
        return 5;
    }
    std::printf("bridge ready in %.0f ms\n", timer.nsecsElapsed() / 1e6);
    FrameGen fg(rig.log);
    if (with_fg && !fg.ensureInitialized(rig.vk, FrameGen::Backend::Dlss12)) {
        std::printf("VERDICT: DLSS FG D3D12 could not start next to neural rendering\n");
        nr.destroy();
        return 6;
    }

    int failures = 0, good = 0;
    double sum_ms = 0.0, sum_gpu = 0.0;
    int timed = 0;
    int out_w = w, out_h = h;
    std::vector<uint8_t> input(static_cast<size_t>(w) * h * 4);
    std::vector<float> reference, output, last_full;
    struct pl_frame last_out = {};
    for (int k = 0; k < frames; k++) {
        const int shift = mvx * k;
        if (base.empty()) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    uint8_t *p = input.data() + (static_cast<size_t>(y) * w + x) * 4;
                    p[0] = p[1] = p[2] = pattern(x, y, shift);
                    p[3] = 255;
                }
            }
        } else {
            const int s = ((shift % w) + w) % w;
            for (int y = 0; y < h; y++) {
                const uint8_t *src_row = base.data() + static_cast<size_t>(y) * w * 4;
                uint8_t *dst_row = input.data() + static_cast<size_t>(y) * w * 4;
                std::memcpy(dst_row, src_row + static_cast<size_t>(s) * 4, static_cast<size_t>(w - s) * 4);
                std::memcpy(dst_row + static_cast<size_t>(w - s) * 4, src_row, static_cast<size_t>(s) * 4);
            }
        }
        struct pl_frame src;
        if (hdr) {
            const std::vector<uint16_t> half = toHdrHalf(input, rig.hdr_peak);
            rig.uploadHalf(half.data());
            src = rig.hdrSourceFrame();
        } else {
            rig.uploadRgba(input.data());
            src = rig.sourceFrame();
        }
        // --tune-at: the controls change while running (no new feature, a new history).
        if (tune_at >= 0 && k == tune_at) {
            for (auto &t : options.tuning) {
                t.style = 2;
                t.local_tone = 2.0f;
            }
        }
        struct pl_frame out = {};
        uint64_t signature = 0;
        timer.restart();
        bool ok;
        std::string pipe_note;
        if (pipeline) {
            ok = nr.begin(&src, static_cast<uint64_t>(k + 1), options);
            const double begin_ms = timer.nsecsElapsed() / 1e6;
            const double wait_ms = ok ? waitBegun([&]() { return nr.beginDone(); }) : -1.0;
            ok = ok && wait_ms >= 0.0 && nr.process(&src, static_cast<uint64_t>(k + 1), options, &out, &signature);
            char buf[96];
            std::snprintf(buf, sizeof(buf), " | begin %.1f ms, D3D12 done after %.1f ms", begin_ms, wait_ms);
            pipe_note = buf;
        } else {
            ok = nr.process(&src, static_cast<uint64_t>(k + 1), options, &out, &signature);
        }
        const double ms = timer.nsecsElapsed() / 1e6;
        if (!ok) {
            std::printf("frame %2d: process() failed (%.1f ms)\n", k, ms);
            failures++;
            continue;
        }
        out_w = static_cast<int>(std::lround(std::fabs(pl_rect_w(out.crop))));
        out_h = static_cast<int>(std::lround(std::fabs(pl_rect_h(out.crop))));
        // Both seen the same way (SDR BT.709, HDR tone mapped by libplacebo, the output scaled
        // back to the input size): the output must stay close to its input.
        if (!rig.readFloat(src, w, h, pl_color_space_bt709, reference) ||
            !rig.readFloat(out, w, h, pl_color_space_bt709, output)) {
            std::printf("frame %2d: reading the output failed\n", k);
            failures++;
            continue;
        }
        if (k >= 2) {
            sum_ms += ms;
            sum_gpu += nr.gpuTimeMs();
            timed++;
        }
        double diff = 0.0, mean = 0.0;
        for (size_t i = 0; i < output.size(); i += 4) {
            for (int c = 0; c < 3; c++) {
                diff += std::fabs(static_cast<double>(output[i + c]) - static_cast<double>(reference[i + c])) * 255.0;
                mean += output[i + c] * 255.0;
            }
        }
        const double n = static_cast<double>(output.size() / 4) * 3.0;
        diff /= n;
        mean /= n;
        bool frame_good = mean > 5.0 && diff < 40.0;
        if (options.out_width && (out_w != options.out_width || out_h != options.out_height))
            frame_good = false;
        std::string hdr_note;
        if (hdr) {
            std::vector<float> lin_in, lin_out;
            if (rig.readFloat(src, w, h, rig.hdrSpace(), lin_in) &&
                rig.readFloat(out, out_w, out_h, rig.hdrSpace(), lin_out)) {
                const HdrStats a = hdrStats(lin_in), b = hdrStats(lin_out);
                char buf[200];
                std::snprintf(buf, sizeof(buf), " | HDR in: mean %.3f max %.2f >1.0 %.1f%%, out: mean %.3f max %.2f >1.0 %.1f%%",
                              a.mean, a.max, a.above, b.mean, b.max, b.above);
                hdr_note = buf;
                // The highlights must survive (not clipped at SDR white).
                if (b.max < 2.0 || b.above < a.above * 0.5)
                    frame_good = false;
            }
        }
        std::string fg_note;
        if (with_fg) {
            std::vector<struct pl_frame> fg_out;
            bool fg_ok;
            if (pipeline) {
                fg_ok = fg.begin(&out, out_w, out_h, 2, 16.7f) && waitBegun([&]() { return fg.beginDone(); }) >= 0.0 &&
                        fg.finish(fg_out);
            } else {
                fg_ok = fg.process(&out, out_w, out_h, 2, 16.7f, fg_out);
            }
            fg_note = fg_ok ? " | FG " + std::to_string(fg_out.size()) + " frame(s)" : " | FG failed";
            if (!fg_ok && k > 0)
                failures++;
        }
        std::printf("frame %2d: %dx%d, process %.1f ms, NR GPU %.2f ms, mean diff to input %.2f, output mean %.1f%s%s%s %s\n",
                    k, out_w, out_h, ms, nr.gpuTimeMs(), diff, mean, pipe_note.c_str(), hdr_note.c_str(),
                    fg_note.c_str(), frame_good ? "OK" : "BAD");
        if (frame_good)
            good++;
        else
            failures++;
        last_out = out;
    }
    if (!out_path.empty() && last_out.num_planes > 0) {
        // The last output at its own size, as an SDR picture (HDR tone mapped by libplacebo).
        if (rig.readFloat(last_out, out_w, out_h, pl_color_space_bt709, last_full)) {
            std::vector<uint8_t> bytes(last_full.size());
            for (size_t i = 0; i < last_full.size(); i++)
                bytes[i] = static_cast<uint8_t>(std::clamp(last_full[i], 0.0f, 1.0f) * 255.0f + 0.5f);
            QImage saved(bytes.data(), out_w, out_h, out_w * 4, QImage::Format_RGBA8888);
            std::printf("%s %s (%dx%d)\n", saved.copy().save(QString::fromStdString(out_path)) ? "saved" : "could not save",
                        out_path.c_str(), out_w, out_h);
        }
    }
    if (timed)
        std::printf("-- %d of %d frames good, %dx%d -> %dx%d %s, average process() %.2f ms, NR GPU time %.2f ms\n", good,
                    frames, w, h, out_w, out_h, hdr ? "HDR" : "SDR", sum_ms / timed, sum_gpu / timed);
    fg.destroy();
    nr.destroy();
    std::printf("\nVERDICT: %s\n", failures == 0 ? "NEURAL RENDERING PASSED" : "NEURAL RENDERING FAILED, see above");
    return failures == 0 ? 0 : 1;
}

} // namespace

// ---- --eval: frame generation against held-out real frames ----
// A 60 fps capture as PNG files (8-bit: SDR; 16-bit: HDR, PQ BT.2020 values as decoded, no tone
// mapping). Every `multiplier`-th frame goes through FrameGen as a real frame, so the generated ones
// sit where the frames in between were: those are the ground truth. Every generated frame is
// compared with it, and so are two things frame generation must beat: the last real frame shown
// again (what the display shows without it) and a plain blend of the two real frames.
struct EvalPlane
{
    int w = 0, h = 0;
    std::vector<float> y; // luma of the encoded values (SDR: gamma, HDR: PQ), 0..255
    std::vector<float> rgb; // encoded values 0..255, 3 per pixel (for the pictures saved)
};

struct EvalScore
{
    double mae = 0.0;   // mean absolute luma difference to the truth (0..255)
    double psnr = 0.0;  // luma PSNR (dB)
    double art = 0.0;   // % of 16x16 blocks whose mean difference is above 12: visible errors
    double worse = 0.0; // % of blocks clearly worse than showing the last real frame again
    std::vector<float> blocks; // mean difference per block
};

bool evalLoad(const QString &path, bool want_hdr, EvalPlane &plane, std::vector<uint8_t> *rgba8, std::vector<uint16_t> *rgba16)
{
    QImage img(path);
    if (img.isNull())
        return false;
    plane.w = img.width();
    plane.h = img.height();
    const size_t n = static_cast<size_t>(plane.w) * plane.h;
    plane.y.resize(n);
    plane.rgb.resize(n * 3);
    if (want_hdr) {
        img = img.convertToFormat(QImage::Format_RGBA64);
        if (rgba16)
            rgba16->resize(n * 4);
        for (int yy = 0; yy < plane.h; yy++) {
            const uint16_t *row = reinterpret_cast<const uint16_t *>(img.constScanLine(yy));
            for (int x = 0; x < plane.w; x++) {
                const size_t i = static_cast<size_t>(yy) * plane.w + x;
                const float r = row[4 * x] / 257.0f, g = row[4 * x + 1] / 257.0f, b = row[4 * x + 2] / 257.0f;
                plane.rgb[3 * i] = r;
                plane.rgb[3 * i + 1] = g;
                plane.rgb[3 * i + 2] = b;
                plane.y[i] = 0.2627f * r + 0.6780f * g + 0.0593f * b;
                if (rgba16) {
                    std::memcpy(rgba16->data() + 4 * i, row + 4 * x, 3 * sizeof(uint16_t));
                    (*rgba16)[4 * i + 3] = 0xffff;
                }
            }
        }
    } else {
        img = img.convertToFormat(QImage::Format_RGBA8888);
        if (rgba8)
            rgba8->resize(n * 4);
        for (int yy = 0; yy < plane.h; yy++) {
            const uint8_t *row = img.constScanLine(yy);
            for (int x = 0; x < plane.w; x++) {
                const size_t i = static_cast<size_t>(yy) * plane.w + x;
                const float r = row[4 * x], g = row[4 * x + 1], b = row[4 * x + 2];
                plane.rgb[3 * i] = r;
                plane.rgb[3 * i + 1] = g;
                plane.rgb[3 * i + 2] = b;
                plane.y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                if (rgba8) {
                    std::memcpy(rgba8->data() + 4 * i, row + 4 * x, 3);
                    (*rgba8)[4 * i + 3] = 255;
                }
            }
        }
    }
    return true;
}

// Compares `got` with `truth`; `repeat` (the last real frame) only for the `worse` count.
EvalScore evalScore(const EvalPlane &got, const EvalPlane &truth, const EvalScore *repeat)
{
    EvalScore s;
    const int w = truth.w, h = truth.h;
    double sum = 0.0, sq = 0.0;
    const int bw = (w + 15) / 16, bh = (h + 15) / 16;
    std::vector<double> bsum(static_cast<size_t>(bw) * bh, 0.0);
    std::vector<int> bcount(static_cast<size_t>(bw) * bh, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const size_t i = static_cast<size_t>(y) * w + x;
            const double d = std::fabs(static_cast<double>(got.y[i]) - truth.y[i]);
            sum += d;
            sq += d * d;
            const size_t b = static_cast<size_t>(y / 16) * bw + x / 16;
            bsum[b] += d;
            bcount[b]++;
        }
    }
    const double n = static_cast<double>(w) * h;
    s.mae = sum / n;
    const double mse = sq / n;
    s.psnr = mse > 1e-9 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
    s.blocks.resize(bsum.size());
    int art = 0, worse = 0;
    for (size_t b = 0; b < bsum.size(); b++) {
        const float m = static_cast<float>(bsum[b] / std::max(bcount[b], 1));
        s.blocks[b] = m;
        if (m > 12.0f)
            art++;
        if (repeat && m > 6.0f && m > repeat->blocks[b] + 4.0f)
            worse++;
    }
    s.art = 100.0 * art / std::max<size_t>(bsum.size(), 1);
    s.worse = 100.0 * worse / std::max<size_t>(bsum.size(), 1);
    return s;
}

// truth | generated | difference (16x16 block error in red, x4), side by side, 8-bit.
void evalSave(const std::string &dir, const std::string &name, const EvalPlane &truth, const EvalPlane &got,
              const EvalScore &score)
{
    const int w = truth.w, h = truth.h, bw = (w + 15) / 16;
    QImage out(w * 3, h, QImage::Format_RGB888);
    for (int y = 0; y < h; y++) {
        uchar *row = out.scanLine(y);
        for (int x = 0; x < w; x++) {
            const size_t i = static_cast<size_t>(y) * w + x;
            for (int c = 0; c < 3; c++) {
                row[3 * x + c] = static_cast<uchar>(std::clamp(truth.rgb[3 * i + c], 0.0f, 255.0f));
                row[3 * (w + x) + c] = static_cast<uchar>(std::clamp(got.rgb[3 * i + c], 0.0f, 255.0f));
            }
            const float d = std::fabs(got.y[i] - truth.y[i]);
            const float blk = score.blocks[static_cast<size_t>(y / 16) * bw + x / 16];
            const uchar base = static_cast<uchar>(std::clamp(truth.y[i] * 0.25f, 0.0f, 255.0f));
            row[3 * (2 * w + x)] = static_cast<uchar>(std::clamp(base + d * 4.0f + (blk > 12.0f ? 60.0f : 0.0f), 0.0f, 255.0f));
            row[3 * (2 * w + x) + 1] = base;
            row[3 * (2 * w + x) + 2] = base;
        }
    }
    out.save(QString::fromStdString(dir + "/" + name + ".png"));
}

int runEval(Rig &rig, const std::string &dir, const std::string &out_dir, FrameGen::Backend backend, int multiplier,
            int cleanup, int dlss_motion, bool flow_hq, bool force_sdr, int start, int count, double fps, int save_worst)
{
    QDir d(QString::fromStdString(dir));
    QStringList files = d.entryList(QStringList() << "*.png", QDir::Files, QDir::Name);
    if (files.size() < multiplier + 1) {
        std::printf("VERDICT: --eval needs at least %d PNG frames in %s (found %lld)\n", multiplier + 1, dir.c_str(),
                    static_cast<long long>(files.size()));
        return 2;
    }
    EvalPlane probe;
    QImage first(d.filePath(files[0]));
    const bool hdr = !force_sdr && first.depth() == 64;
    if (!evalLoad(d.filePath(files[0]), hdr, probe, nullptr, nullptr)) {
        std::printf("VERDICT: cannot read %s\n", qPrintable(files[0]));
        return 2;
    }
    const int w = probe.w, h = probe.h;
    const double interval_ms = 1000.0 / fps * multiplier;
    std::printf("eval: %lld frames %dx%d %s in %s; every %dth frame real (%.1f ms apart), %s, clean-up %d, DLSS motion %d%s\n",
                static_cast<long long>(files.size()), w, h, hdr ? "HDR (PQ BT.2020)" : "SDR", dir.c_str(), multiplier,
                interval_ms, FrameGen::backendName(backend), cleanup, dlss_motion, flow_hq ? ", Flow 1/2" : "");
    if (!rig.resize(w, h)) {
        std::printf("VERDICT: textures\n");
        return 3;
    }
    pl_tex src16 = nullptr;
    struct pl_color_space pq = {};
    pq.primaries = PL_COLOR_PRIM_BT_2020;
    pq.transfer = PL_COLOR_TRC_PQ;
    pq.hdr.max_luma = 10000.0f;
    pq.hdr.min_luma = 0.0f;
    pq.hdr.prim = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020);
    if (hdr) {
        struct pl_tex_params p = {};
        p.w = w;
        p.h = h;
        p.format = pl_find_named_fmt(rig.gpu, "rgba16");
        p.sampleable = true;
        p.host_writable = true;
        src16 = p.format ? pl_tex_create(rig.gpu, &p) : nullptr;
        if (!src16) {
            std::printf("VERDICT: no rgba16 texture\n");
            return 3;
        }
    }
    FrameGen fg(rig.log);
    fg.setCleanup(cleanup);
    fg.setDlssMotion(dlss_motion);
    fg.setFlowHighQuality(flow_hq);
    if (!fg.ensureInitialized(rig.vk, backend)) {
        std::printf("VERDICT: backend could not start (see the log above)\n");
        if (src16)
            pl_tex_destroy(rig.gpu, &src16);
        return 3;
    }
    struct Worst
    {
        double art;
        int index;
    };
    std::vector<Worst> worst;
    double sum_fg_mae = 0, sum_fg_psnr = 0, sum_fg_art = 0, sum_fg_worse = 0, sum_rep_mae = 0, sum_rep_psnr = 0,
           sum_rep_art = 0, sum_blend_mae = 0, sum_blend_psnr = 0, sum_blend_art = 0, sum_real_mae = 0;
    int scored = 0, cuts = 0, failures = 0, warmup = 6, real_reads = 0;
    EvalPlane prev;
    bool have_prev = false;
    const int last = count > 0 ? std::min<int>(files.size() - 1, start + count) : static_cast<int>(files.size()) - 1;
    for (int k = start; k <= last; k += multiplier) {
        EvalPlane real;
        std::vector<uint8_t> rgba8;
        std::vector<uint16_t> rgba16;
        if (!evalLoad(d.filePath(files[k]), hdr, real, hdr ? nullptr : &rgba8, hdr ? &rgba16 : nullptr) ||
            real.w != w || real.h != h) {
            std::printf("frame %d: cannot read %s or another size\n", k, qPrintable(files[k]));
            failures++;
            break;
        }
        // A scene cut: chiaki-ng's duplicate-frame detector resets frame generation there.
        bool cut = false;
        if (have_prev) {
            EvalScore jump = evalScore(prev, real, nullptr);
            if (jump.mae > 30.0) {
                cut = true;
                cuts++;
                fg.reset();
            }
        }
        struct pl_frame src = rig.sourceFrame();
        if (hdr) {
            struct pl_tex_transfer_params up = {};
            up.tex = src16;
            up.ptr = rgba16.data();
            up.row_pitch = static_cast<size_t>(w) * 8;
            pl_tex_upload(rig.gpu, &up);
            src.planes[0].texture = src16;
            src.color = pq;
        } else {
            rig.uploadRgba(rgba8.data());
        }
        std::vector<struct pl_frame> outputs;
        if (!fg.process(&src, w, h, multiplier, static_cast<float>(interval_ms), outputs) || outputs.empty()) {
            std::printf("frame %d: process() failed\n", k);
            failures++;
            prev = std::move(real);
            have_prev = true;
            continue;
        }
        auto readPlane = [&](const struct pl_frame &f, EvalPlane &p) {
            p.w = w;
            p.h = h;
            const size_t n = static_cast<size_t>(w) * h;
            p.y.resize(n);
            p.rgb.resize(n * 3);
            if (hdr) {
                std::vector<float> fl;
                if (!rig.readFloat(f, w, h, pq, fl))
                    return false;
                for (size_t i = 0; i < n; i++) {
                    const float r = fl[4 * i] * 255.0f, g = fl[4 * i + 1] * 255.0f, b = fl[4 * i + 2] * 255.0f;
                    p.rgb[3 * i] = r;
                    p.rgb[3 * i + 1] = g;
                    p.rgb[3 * i + 2] = b;
                    p.y[i] = 0.2627f * r + 0.6780f * g + 0.0593f * b;
                }
            } else {
                std::vector<uint8_t> rgba;
                if (!rig.readRgba(f, rgba))
                    return false;
                for (size_t i = 0; i < n; i++) {
                    const float r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
                    p.rgb[3 * i] = r;
                    p.rgb[3 * i + 1] = g;
                    p.rgb[3 * i + 2] = b;
                    p.y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                }
            }
            return true;
        };
        // The real frame frame generation hands back: must be the source as it was.
        EvalPlane back;
        if (readPlane(outputs.back(), back)) {
            sum_real_mae += evalScore(back, real, nullptr).mae;
            real_reads++;
        }
        const size_t generated = outputs.size() - 1;
        if (have_prev && !cut && generated == static_cast<size_t>(multiplier - 1) && warmup <= 0) {
            for (size_t j = 0; j < generated; j++) {
                const int ti = k - multiplier + 1 + static_cast<int>(j);
                EvalPlane truth, got;
                if (!evalLoad(d.filePath(files[ti]), hdr, truth, nullptr, nullptr) || !readPlane(outputs[j], got)) {
                    std::printf("frame %d: reading failed\n", ti);
                    failures++;
                    continue;
                }
                const float t = static_cast<float>(j + 1) / multiplier;
                EvalPlane blend = truth;
                for (size_t i = 0; i < blend.y.size(); i++)
                    blend.y[i] = prev.y[i] * (1.0f - t) + real.y[i] * t;
                const EvalScore rep = evalScore(prev, truth, nullptr);
                const EvalScore bl = evalScore(blend, truth, nullptr);
                const EvalScore s = evalScore(got, truth, &rep);
                std::printf("frame %5d (t=%.2f): FG psnr %5.2f mae %5.2f art %5.2f%% worse %5.2f%% | repeat psnr %5.2f art %5.2f%% "
                            "| blend psnr %5.2f art %5.2f%%\n",
                            ti, t, s.psnr, s.mae, s.art, s.worse, rep.psnr, rep.art, bl.psnr, bl.art);
                std::fflush(stdout);
                sum_fg_mae += s.mae;
                sum_fg_psnr += s.psnr;
                sum_fg_art += s.art;
                sum_fg_worse += s.worse;
                sum_rep_mae += rep.mae;
                sum_rep_psnr += rep.psnr;
                sum_rep_art += rep.art;
                sum_blend_mae += bl.mae;
                sum_blend_psnr += bl.psnr;
                sum_blend_art += bl.art;
                scored++;
                worst.push_back({ s.art + s.worse, ti });
                if (!out_dir.empty() && save_worst > 0) {
                    // Keep the pictures of the worst frames so far (written now: the planes are gone later).
                    std::sort(worst.begin(), worst.end(), [](const Worst &a, const Worst &b) { return a.art > b.art; });
                    const bool among = std::any_of(worst.begin(), worst.begin() + std::min<size_t>(worst.size(), save_worst),
                                                   [&](const Worst &x) { return x.index == ti; });
                    if (among) {
                        char name[64];
                        std::snprintf(name, sizeof(name), "frame%05d", ti);
                        evalSave(out_dir, name, truth, got, s);
                    }
                }
            }
        }
        if (warmup > 0)
            warmup--;
        prev = std::move(real);
        have_prev = true;
    }
    fg.destroy();
    if (src16)
        pl_tex_destroy(rig.gpu, &src16);
    if (scored == 0) {
        std::printf("VERDICT: nothing scored (%d failures)\n", failures);
        return 1;
    }
    std::sort(worst.begin(), worst.end(), [](const Worst &a, const Worst &b) { return a.art > b.art; });
    if (!out_dir.empty() && save_worst > 0) {
        // Pictures were written for frames that were among the worst at the time: keep the final ones.
        QDir od(QString::fromStdString(out_dir));
        for (const QString &f : od.entryList(QStringList() << "frame*.png", QDir::Files)) {
            const int index = f.mid(5, 5).toInt();
            const bool keep = std::any_of(worst.begin(), worst.begin() + std::min<size_t>(worst.size(), save_worst),
                                          [&](const Worst &x) { return x.index == index; });
            if (!keep)
                od.remove(f);
        }
    }
    std::printf("\n-- %d generated frames scored, %d scene cuts skipped, real frames handed back with mae %.3f\n", scored, cuts,
                sum_real_mae / std::max(1, real_reads));
    std::printf("-- FG      psnr %.2f dB, mae %.2f, art %.2f%% of blocks, worse than repeating %.2f%%\n", sum_fg_psnr / scored,
                sum_fg_mae / scored, sum_fg_art / scored, sum_fg_worse / scored);
    std::printf("-- repeat  psnr %.2f dB, mae %.2f, art %.2f%%\n", sum_rep_psnr / scored, sum_rep_mae / scored, sum_rep_art / scored);
    std::printf("-- blend   psnr %.2f dB, mae %.2f, art %.2f%%\n", sum_blend_psnr / scored, sum_blend_mae / scored,
                sum_blend_art / scored);
    std::printf("-- worst frames (art + worse):");
    for (size_t i = 0; i < std::min<size_t>(worst.size(), 8); i++)
        std::printf(" %d (%.1f)", worst[i].index, worst[i].art);
    std::printf("\nVERDICT: EVAL DONE%s\n", failures ? " (with failures, see above)" : "");
    return failures ? 1 : 0;
}

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
#if defined(_WIN32)
    // 1 ms timer resolution, as chiaki-ng sets it while streaming (windowsStreamTuning.cpp): waits
    // for Streamline's frames sleep 1 ms at a time.
    timeBeginPeriod(1);
#endif
    int backend_value = 5, frames = 20, mvx = 8, w = 1920, h = 1080, model = 0, passes = 1, tune_at = -1;
    bool nr_mode = false, with_fg = false, hdr = false, pipeline = false, timing = false;
    float scale = 1.0f;
    std::string image_path, out_path;
    // --eval
    std::string eval_dir;
    int eval_multiplier = 2, eval_cleanup = 2, eval_motion = 1, eval_start = 0, eval_count = 0, eval_save = 6;
    bool eval_flow_hq = false, eval_sdr = false;
    double eval_fps = 60.0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend_value = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (a == "--mvx" && i + 1 < argc) mvx = std::atoi(argv[++i]);
        else if (a == "--w" && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) h = std::atoi(argv[++i]);
        else if (a == "--nr") nr_mode = true;
        else if (a == "--with-fg") with_fg = true;
        else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--scale" && i + 1 < argc) scale = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--hdr") hdr = true;
        else if (a == "--model" && i + 1 < argc) model = std::atoi(argv[++i]);
        else if (a == "--passes" && i + 1 < argc) passes = std::atoi(argv[++i]);
        else if (a == "--tune-at" && i + 1 < argc) tune_at = std::atoi(argv[++i]);
        else if (a == "--pipeline") pipeline = true;
        else if (a == "--timing") timing = true;
        else if (a == "--eval" && i + 1 < argc) eval_dir = argv[++i];
        else if (a == "--multiplier" && i + 1 < argc) eval_multiplier = std::clamp(std::atoi(argv[++i]), 2, 4);
        else if (a == "--cleanup" && i + 1 < argc) eval_cleanup = std::clamp(std::atoi(argv[++i]), 0, 2);
        else if (a == "--dlss-motion" && i + 1 < argc) eval_motion = std::clamp(std::atoi(argv[++i]), 0, 2);
        else if (a == "--flow-hq") eval_flow_hq = true;
        else if (a == "--sdr") eval_sdr = true;
        else if (a == "--start" && i + 1 < argc) eval_start = std::max(0, std::atoi(argv[++i]));
        else if (a == "--count" && i + 1 < argc) eval_count = std::max(0, std::atoi(argv[++i]));
        else if (a == "--fps" && i + 1 < argc) eval_fps = std::max(1.0, std::atof(argv[++i]));
        else if (a == "--save-worst" && i + 1 < argc) eval_save = std::max(0, std::atoi(argv[++i]));
    }
    if (!eval_dir.empty()) {
        const auto eval_backend = static_cast<FrameGen::Backend>(backend_value);
        if (!FrameGen::backendCompiledIn(eval_backend)) {
            std::printf("VERDICT: backend not compiled in\n");
            return 2;
        }
        Rig rig;
        if (!rig.init()) {
            std::printf("VERDICT: could not create the libplacebo Vulkan device\n");
            rig.destroy();
            return 3;
        }
        if (!out_path.empty())
            QDir().mkpath(QString::fromStdString(out_path));
        const int result = runEval(rig, eval_dir, out_path, eval_backend, eval_multiplier, eval_cleanup, eval_motion,
                                   eval_flow_hq, eval_sdr, eval_start, eval_count, eval_fps, eval_save);
        rig.destroy();
        return result;
    }
    if (nr_mode) {
        if (!NeuralRender::compiledIn()) {
            std::printf("VERDICT: neural rendering not compiled in\n");
            return 2;
        }
        Rig rig;
        if (!rig.init()) {
            std::printf("VERDICT: could not create the libplacebo Vulkan device\n");
            rig.destroy();
            return 3;
        }
        const int result =
            runNeuralRender(rig, image_path, out_path, frames, mvx, w, h, with_fg, scale, hdr, model, passes, tune_at,
                            pipeline);
        rig.destroy();
        return result;
    }
    const auto backend = static_cast<FrameGen::Backend>(backend_value);
    std::printf("chiaki-fgdx12test: backend %s, %d frames, %d px/frame motion%s%s%s\n", FrameGen::backendName(backend), frames,
                mvx, pipeline ? ", pipelined (begin / beginDone / finish)" : "",
                hdr ? ", HDR (PQ BT.2020 RGBA16F, up to 1500 nits)" : "",
                timing ? ", timing only (paced at 60 fps, frames not checked)" : "");
    // Frame generation must stay off for HDR frames: an 8-bit backend, or (test hook) a runtime
    // that refuses RGBA16F, which must not take the backend down for SDR frames either.
    const bool refuse_hdr = hdr && std::getenv("CHIAKI_DX12FG_REFUSE_HDR") != nullptr;
    const bool hdr_off = hdr && (!FrameGen::backendHdrCapable(backend) || refuse_hdr);
    if (hdr && !FrameGen::backendHdrCapable(backend))
        std::printf("note: %s has no HDR path, frame generation must stay off for HDR frames\n", FrameGen::backendName(backend));
    if (refuse_hdr)
        std::printf("note: CHIAKI_DX12FG_REFUSE_HDR: the runtime refuses HDR, frame generation must stay off for HDR frames\n");
    if (!FrameGen::backendCompiledIn(backend)) {
        std::printf("VERDICT: backend not compiled in\n");
        return 2;
    }

    Rig rig;
    if (!rig.init()) {
        std::printf("VERDICT: could not create the libplacebo Vulkan device\n");
        rig.destroy();
        return 3;
    }
    {
        auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            rig.inst->get_proc_addr(rig.inst->instance, "vkGetPhysicalDeviceProperties"));
        if (get_props) {
            VkPhysicalDeviceProperties props = {};
            get_props(rig.vk->phys_device, &props);
            std::printf("vulkan device: %s\n", props.deviceName);
        }
    }

    const Phase phases[] = {
        { "2x, clean-up off", w, h, 2, frames, 0 },
        { "2x, clean-up on (default)", w, h, 2, std::min(frames, 8), 2 },
        { "4x, clean-up off", w, h, 4, std::min(frames, 8), 0 },
        { "4x, clean-up on (default)", w, h, 4, std::min(frames, 8), 2 },
        { "resize to 1280x720, 2x", 1280, 720, 2, std::min(frames, 6), 0 },
        { "back to the first size, 3x", w, h, 3, std::min(frames, 6), 0 },
    };

    FrameGen fg(rig.log);
    fg.setCleanup(0);
    int failures = 0;
    int shift_counter = 0;
    for (const Phase &phase : phases) {
        std::printf("\n== %s (%dx%d)\n", phase.name, phase.w, phase.h);
        if (!rig.resize(phase.w, phase.h)) {
            std::printf("FAIL: textures\n");
            failures++;
            break;
        }
        fg.setCleanup(phase.cleanup);
        if (!fg.ensureInitialized(rig.vk, backend)) {
            std::printf("FAIL: backend could not start (see the log above)\n");
            failures++;
            break;
        }
        int good = 0, checked = 0;
        double sum_ms = 0.0;
        int timed = 0;
        std::vector<double> times; // --timing
        int timing_failures = 0;
        PqStats hdr_src = {};
        for (int k = 0; k < phase.frames; k++) {
            shift_counter++;
            struct pl_frame src;
            if (hdr) {
                if (!rig.uploadPq(mvx * shift_counter)) {
                    std::printf("FAIL: HDR source texture\n");
                    failures++;
                    break;
                }
                src = rig.pqSourceFrame();
                std::vector<float> pq, red;
                if (k == 0 && rig.readPq(src, pq, red))
                    hdr_src = pqStats(pq);
            } else {
                rig.upload(mvx * shift_counter);
                src = rig.sourceFrame();
            }
            std::vector<struct pl_frame> outputs;
            QElapsedTimer timer;
            timer.start();
            bool ok;
            if (pipeline) {
                ok = fg.begin(&src, phase.w, phase.h, phase.multiplier, 33.3f) &&
                     waitBegun([&]() { return fg.beginDone(); }) >= 0.0 && fg.finish(outputs);
            } else {
                ok = fg.process(&src, phase.w, phase.h, phase.multiplier, 33.3f, outputs);
            }
            const double ms = timer.nsecsElapsed() / 1e6;
            if (k >= 2) {
                sum_ms += ms;
                timed++;
            }
            if (timing) {
                if (k > 0 && !hdr_off && (!ok || outputs.empty()))
                    timing_failures++;
                if (k >= 2)
                    times.push_back(ms);
                // The next stream frame comes 16.7 ms after this one started.
                const double left_ms = 1000.0 / 60.0 - timer.nsecsElapsed() / 1e6;
                if (left_ms > 0.0)
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(left_ms * 1000.0)));
                continue;
            }
            if (!ok || outputs.empty()) {
                std::printf("frame %2d: process() returned %s, %zu outputs\n", k, ok ? "true" : "false", outputs.size());
                if (k > 0 && !hdr_off)
                    failures++;
                continue;
            }
            if (hdr_off) {
                std::printf("frame %2d: FAIL, frames came back for an HDR source that must get none\n", k);
                failures++;
                continue;
            }
            const size_t generated = outputs.size() - 1;
            if (k == 0 || generated == 0) {
                std::printf("frame %2d: %zu output(s), process %.2f ms (first frame of a sequence)\n", k, outputs.size(), ms);
                continue;
            }
            // Generated frame i of `generated` sits at t = (i + 1) / (generated + 1) between the
            // previous and the current frame: that is mvx * (1 - t) pixels behind the current one.
            std::string line;
            bool frame_good = true;
            for (size_t i = 0; i < generated; i++) {
                std::vector<uint8_t> red;
                std::vector<float> red_hdr, pq;
                if (hdr ? !rig.readPq(outputs[i], pq, red_hdr) : !rig.read(outputs[i], red)) {
                    std::printf("frame %2d: reading output %zu failed\n", k, i);
                    frame_good = false;
                    break;
                }
                const double t = static_cast<double>(i + 1) / static_cast<double>(generated + 1);
                const int ideal = static_cast<int>(std::lround(mvx * (1.0 - t)));
                double diff_best = 0.0, diff_zero = 0.0;
                const int best = hdr ? bestShift(red_hdr, phase.w, phase.h, mvx * shift_counter, diff_best, diff_zero)
                                     : bestShift(red, phase.w, phase.h, mvx * shift_counter, diff_best, diff_zero);
                char buf[256];
                std::snprintf(buf, sizeof(buf), " [t=%.2f shift %d (ideal %d) diff %.2f vs %.2f]", t, best, ideal, diff_best, diff_zero);
                line += buf;
                if (std::abs(best - ideal) > 1 || diff_best > diff_zero * 0.5)
                    frame_good = false;
                if (hdr) {
                    // The highlights stay (no clipping at SDR white) and the values keep more
                    // than 8 bits (an 8-bit stage would put them on the 8-bit grid).
                    const PqStats g = pqStats(pq);
                    std::snprintf(buf, sizeof(buf), "{peak %.0f nits, >SDR white %.1f%%, grid %.3f}", g.peak_nits, g.above_sdr,
                                  g.grid);
                    line += buf;
                    if (g.peak_nits < hdr_src.peak_nits * 0.8 || g.above_sdr < hdr_src.above_sdr * 0.8 || g.grid < 0.1)
                        frame_good = false;
                }
            }
            std::vector<uint8_t> real;
            std::vector<float> real_hdr, real_pq;
            double real_best = 0.0, real_zero = 0.0;
            int real_shift = 99;
            if (hdr ? rig.readPq(outputs.back(), real_pq, real_hdr) : rig.read(outputs.back(), real))
                real_shift = hdr ? bestShift(real_hdr, phase.w, phase.h, mvx * shift_counter, real_best, real_zero)
                                 : bestShift(real, phase.w, phase.h, mvx * shift_counter, real_best, real_zero);
            if (real_shift != 0)
                frame_good = false;
            if (hdr) {
                // The real frame is only re-encoded: as precise as the source.
                const PqStats r = pqStats(real_pq);
                char buf[128];
                std::snprintf(buf, sizeof(buf), " | real {peak %.0f nits, >SDR white %.1f%%, grid %.3f}", r.peak_nits,
                              r.above_sdr, r.grid);
                line += buf;
                if (r.grid < 0.1 || std::fabs(r.peak_nits - hdr_src.peak_nits) > hdr_src.peak_nits * 0.02)
                    frame_good = false;
            }
            std::printf("frame %2d: %zu generated%s | real frame shift %d, process %.2f ms %s\n", k, generated, line.c_str(),
                        real_shift, ms, frame_good ? "OK" : "BAD");
            checked++;
            if (frame_good)
                good++;
        }
        if (timing) {
            std::sort(times.begin(), times.end());
            if (!times.empty()) {
                const auto at = [&](double q) { return times[std::min(times.size() - 1, static_cast<size_t>(q * times.size()))]; };
                std::printf("-- timing: %zu frames, process() median %.2f ms, p90 %.2f ms, max %.2f ms%s\n", times.size(),
                            at(0.5), at(0.9), times.back(), timing_failures ? ", FAIL: calls without frames" : "");
            }
            if (timing_failures)
                failures++;
            continue;
        }
        if (timed)
            std::printf("-- %d of %d checked frames good, average process() %.2f ms, GPU time reported %.2f ms\n", good,
                        checked, sum_ms / timed, fg.gpuTimeMs());
        if (hdr)
            std::printf("-- HDR source: peak %.0f nits, %.1f%% of the pixels above SDR white (scRGB > 1.0), grid %.3f\n",
                        hdr_src.peak_nits, hdr_src.above_sdr, hdr_src.grid);
        if (hdr_off)
            continue; // nothing to check: no frames is the right answer
        if (checked == 0 || good < checked - 1)
            failures++;
    }

    if (refuse_hdr && failures == 0) {
        // After HDR was refused the same backend must still generate SDR frames.
        std::printf("\n== SDR after the HDR refusal, 2x\n");
        int sdr_frames = 0;
        if (rig.resize(w, h) && !fg.backendFailed(backend) && fg.ensureInitialized(rig.vk, backend)) {
            for (int k = 0; k < 4; k++) {
                shift_counter++;
                rig.upload(mvx * shift_counter);
                struct pl_frame src = rig.sourceFrame();
                std::vector<struct pl_frame> outputs;
                if (fg.process(&src, w, h, 2, 33.3f, outputs) && outputs.size() == 2)
                    sdr_frames++;
            }
        }
        std::printf("-- %d of 4 SDR frames with a generated frame%s\n", sdr_frames, sdr_frames >= 3 ? "" : ", FAIL");
        if (sdr_frames < 3)
            failures++;
    }
    fg.destroy();
    rig.destroy();
    std::printf("\nVERDICT: %s\n", failures != 0 ? "SOME PHASES FAILED, see above"
                                   : timing ? "TIMING DONE (frames not checked)" : "ALL PHASES PASSED");
    return failures == 0 ? 0 : 1;
}
