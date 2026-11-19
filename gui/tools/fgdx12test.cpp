// chiaki-fgdx12test: console test of chiaki-ng's frame generation backends, driven the way
// the renderer drives them: synthetic frames of a moving pattern go through FrameGen::process
// on a libplacebo Vulkan device, and the generated frames are read back and compared with
// where the pattern should be between the previous and the current real frame.
//
// Usage: chiaki-fgdx12test [--backend 5] [--frames N] [--mvx N] [--w W --h H]
//   --backend  FrameGen::Backend value: 1 DLSS (Vulkan), 3 Flow, 5 DLSS through D3D12 (default)
// For backend 5 on a GPU without NVIDIA's runtime, NeuralHub's dxgi.dll and neuralhub.ini
// must sit next to the executable.

#include "framegen.h"

#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

Q_LOGGING_CATEGORY(chiakiGui, "chiaki.gui", QtInfoMsg)

namespace {

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
        struct pl_vk_inst_params inst_params = {};
        inst = pl_vk_inst_create(log, &inst_params);
        if (!inst)
            return false;
        struct pl_vulkan_params vulkan_params = {
            .instance = inst->instance,
            .get_proc_addr = inst->get_proc_addr,
            .allow_software = false,
            PL_VULKAN_DEFAULTS
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
        struct pl_tex_transfer_params up = {};
        up.tex = source;
        up.ptr = data.data();
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
        std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
        struct pl_tex_transfer_params dn = {};
        dn.tex = readback;
        dn.ptr = rgba.data();
        dn.row_pitch = static_cast<size_t>(w) * 4;
        if (!pl_tex_download(gpu, &dn))
            return false;
        red.resize(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < red.size(); i++)
            red[i] = rgba[4 * i];
        return true;
    }

    void destroy()
    {
        if (gpu) {
            if (source)
                pl_tex_destroy(gpu, &source);
            if (readback)
                pl_tex_destroy(gpu, &readback);
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

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    int backend_value = 5, frames = 20, mvx = 8, w = 1920, h = 1080;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) backend_value = std::atoi(argv[++i]);
        else if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (a == "--mvx" && i + 1 < argc) mvx = std::atoi(argv[++i]);
        else if (a == "--w" && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) h = std::atoi(argv[++i]);
    }
    const auto backend = static_cast<FrameGen::Backend>(backend_value);
    std::printf("chiaki-fgdx12test: backend %s, %d frames, %d px/frame motion\n", FrameGen::backendName(backend), frames, mvx);
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
            const bool ok = fg.process(&src, phase.w, phase.h, phase.multiplier, 33.3f, outputs);
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
