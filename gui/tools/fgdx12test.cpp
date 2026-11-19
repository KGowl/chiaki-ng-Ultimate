// chiaki-fgdx12test: console test of chiaki-ng's frame generation backends, driven the way
// the renderer drives them: synthetic frames of a moving pattern go through FrameGen::process
// on a libplacebo Vulkan device, and the generated frames are read back and compared with
// where the pattern should be between the previous and the current real frame.
//
// Usage: chiaki-fgdx12test [--backend 5] [--frames N] [--mvx N] [--w W --h H]
//   --backend  FrameGen::Backend value: 1 DLSS (Vulkan), 3 Flow, 5 DLSS through D3D12 (default)
//        chiaki-fgdx12test --nr [--image file] [--out file.png] [--with-fg] [--frames N] [--mvx N]
//   --nr       neural rendering (NeuralRender, DLSS 5 through D3D12) on a moving pattern or on
//              an image panned by --mvx px per frame; --out saves the last output, --with-fg
//              also runs DLSS Frame Generation through D3D12 on the output (shared runtime).
// For backend 5 and --nr on a GPU without NVIDIA's runtime, NeuralHub's dxgi.dll and
// neuralhub.ini must sit next to the executable.

#include "framegen.h"
#include "neuralrender.h"

#include <QImage>

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

// Neural rendering on a moving pattern or a panned image: every output must be a real
// picture close to its input (not black, not the previous frame), and the runtime must keep
// working when frame generation through D3D12 shares it.
int runNeuralRender(Rig &rig, const std::string &image_path, const std::string &out_path, int frames, int mvx, int w,
                    int h, bool with_fg)
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
    std::printf("neural rendering test: %dx%d %s, %d frames, %d px/frame pan%s\n", w, h,
                base.empty() ? "pattern" : image_path.c_str(), frames, mvx, with_fg ? ", with DLSS FG D3D12" : "");
    if (!rig.resize(w, h)) {
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
    double sum_ms = 0.0;
    int timed = 0;
    std::vector<uint8_t> input(static_cast<size_t>(w) * h * 4), output;
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
        rig.uploadRgba(input.data());
        struct pl_frame src = rig.sourceFrame();
        struct pl_frame out = {};
        uint64_t signature = 0;
        timer.restart();
        const bool ok = nr.process(&src, static_cast<uint64_t>(k + 1), &out, &signature);
        const double ms = timer.nsecsElapsed() / 1e6;
        if (!ok) {
            std::printf("frame %2d: process() failed (%.1f ms)\n", k, ms);
            failures++;
            continue;
        }
        if (!rig.readRgba(out, output)) {
            std::printf("frame %2d: reading the output failed\n", k);
            failures++;
            continue;
        }
        if (k >= 2) {
            sum_ms += ms;
            timed++;
        }
        double diff = 0.0, mean = 0.0;
        for (size_t i = 0; i < output.size(); i += 4) {
            for (int c = 0; c < 3; c++) {
                diff += std::fabs(static_cast<double>(output[i + c]) - static_cast<double>(input[i + c]));
                mean += output[i + c];
            }
        }
        const double n = static_cast<double>(output.size() / 4) * 3.0;
        diff /= n;
        mean /= n;
        const bool frame_good = mean > 5.0 && diff < 40.0;
        std::string fg_note;
        if (with_fg) {
            std::vector<struct pl_frame> fg_out;
            const bool fg_ok = fg.process(&out, w, h, 2, 16.7f, fg_out);
            fg_note = fg_ok ? " | FG " + std::to_string(fg_out.size()) + " frame(s)" : " | FG failed";
            if (!fg_ok && k > 0)
                failures++;
        }
        std::printf("frame %2d: process %.1f ms, NR GPU %.2f ms, mean diff to input %.2f, output mean %.1f%s %s\n", k, ms,
                    nr.gpuTimeMs(), diff, mean, fg_note.c_str(), frame_good ? "OK" : "BAD");
        if (frame_good)
            good++;
        else
            failures++;
    }
    if (!out_path.empty() && !output.empty()) {
        QImage saved(output.data(), w, h, w * 4, QImage::Format_RGBA8888);
        std::printf("%s %s\n", saved.copy().save(QString::fromStdString(out_path)) ? "saved" : "could not save",
                    out_path.c_str());
    }
    if (timed)
        std::printf("-- %d of %d frames good, average process() %.2f ms, NR GPU time %.2f ms\n", good, frames,
                    sum_ms / timed, nr.gpuTimeMs());
    fg.destroy();
    nr.destroy();
    std::printf("\nVERDICT: %s\n", failures == 0 ? "NEURAL RENDERING PASSED" : "NEURAL RENDERING FAILED, see above");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    int backend_value = 5, frames = 20, mvx = 8, w = 1920, h = 1080;
    bool nr_mode = false, with_fg = false;
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
        const int result = runNeuralRender(rig, image_path, out_path, frames, mvx, w, h, with_fg);
        rig.destroy();
        return result;
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
