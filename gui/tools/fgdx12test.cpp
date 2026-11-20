// chiaki-fgdx12test: console test of chiaki-ng's frame generation backends, driven the way
// the renderer drives them: synthetic frames of a moving pattern go through FrameGen::process
// on a libplacebo Vulkan device, and the generated frames are read back and compared with
// where the pattern should be between the previous and the current real frame.
//
// Usage: chiaki-fgdx12test [--backend 5] [--frames N] [--mvx N] [--w W --h H]
//   --backend  FrameGen::Backend value: 1 DLSS (Vulkan), 3 Flow, 5 DLSS through D3D12 (default)
//        chiaki-fgdx12test --nr [--image file] [--out file.png] [--with-fg] [--frames N] [--mvx N]
//                          [--scale 1|1.5|2] [--hdr] [--model 0|1] [--passes 1-3] [--tune-at K]
//   --nr       neural rendering (NeuralRender, DLSS 5 through D3D12) on a moving pattern or on
//              an image panned by --mvx px per frame; --out saves the last output, --with-fg
//              also runs DLSS Frame Generation through D3D12 on the output (shared runtime),
//              --scale upscales after it, --hdr feeds a linear-light HDR frame, --model picks
//              NVIDIA's feature 18 model (0 CC, 1 HT).
//   --pipeline drives backend 5 and --nr (with --with-fg) the way the renderer pipelines them:
//              begin(), poll beginDone(), then process() / finish(), with the same checks.
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
#include <QElapsedTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <chrono>
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
int bestShift(const std::vector<uint8_t> &red, int w, int h, int shift_px, double &diff_best, double &diff_zero)
{
    double dsum[17] = {};
    int count = 0;
    for (int y = h * 4 / 10; y < h * 6 / 10; y += 4) {
        const uint8_t *row = red.data() + static_cast<size_t>(y) * w;
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

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
#if defined(_WIN32)
    // 1 ms timer resolution, as chiaki-ng sets it while streaming (windowsStreamTuning.cpp): waits
    // for Streamline's frames sleep 1 ms at a time.
    timeBeginPeriod(1);
#endif
    int backend_value = 5, frames = 20, mvx = 8, w = 1920, h = 1080, model = 0, passes = 1, tune_at = -1;
    bool nr_mode = false, with_fg = false, hdr = false, pipeline = false;
    float scale = 1.0f;
    std::string image_path, out_path;
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
    std::printf("chiaki-fgdx12test: backend %s, %d frames, %d px/frame motion%s\n", FrameGen::backendName(backend), frames,
                mvx, pipeline ? ", pipelined (begin / beginDone / finish)" : "");
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
        for (int k = 0; k < phase.frames; k++) {
            shift_counter++;
            rig.upload(mvx * shift_counter);
            struct pl_frame src = rig.sourceFrame();
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
            if (!ok || outputs.empty()) {
                std::printf("frame %2d: process() returned %s, %zu outputs\n", k, ok ? "true" : "false", outputs.size());
                if (k > 0)
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
                if (!rig.read(outputs[i], red)) {
                    std::printf("frame %2d: reading output %zu failed\n", k, i);
                    frame_good = false;
                    break;
                }
                const double t = static_cast<double>(i + 1) / static_cast<double>(generated + 1);
                const int ideal = static_cast<int>(std::lround(mvx * (1.0 - t)));
                double diff_best = 0.0, diff_zero = 0.0;
                const int best = bestShift(red, phase.w, phase.h, mvx * shift_counter, diff_best, diff_zero);
                char buf[160];
                std::snprintf(buf, sizeof(buf), " [t=%.2f shift %d (ideal %d) diff %.2f vs %.2f]", t, best, ideal, diff_best, diff_zero);
                line += buf;
                if (std::abs(best - ideal) > 1 || diff_best > diff_zero * 0.5)
                    frame_good = false;
            }
            std::vector<uint8_t> real;
            double real_best = 0.0, real_zero = 0.0;
            int real_shift = 99;
            if (rig.read(outputs.back(), real))
                real_shift = bestShift(real, phase.w, phase.h, mvx * shift_counter, real_best, real_zero);
            if (real_shift != 0)
                frame_good = false;
            std::printf("frame %2d: %zu generated%s | real frame shift %d, process %.2f ms %s\n", k, generated, line.c_str(),
                        real_shift, ms, frame_good ? "OK" : "BAD");
            checked++;
            if (frame_good)
                good++;
        }
        if (timed)
            std::printf("-- %d of %d checked frames good, average process() %.2f ms, GPU time reported %.2f ms\n", good,
                        checked, sum_ms / timed, fg.gpuTimeMs());
        if (checked == 0 || good < checked - 1)
            failures++;
    }

    fg.destroy();
    rig.destroy();
    std::printf("\nVERDICT: %s\n", failures == 0 ? "ALL PHASES PASSED" : "SOME PHASES FAILED, see above");
    return failures == 0 ? 0 : 1;
}
