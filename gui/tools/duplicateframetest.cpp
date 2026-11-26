// chiaki-duplicateframetest: DuplicateFrameDetector (include/duplicateframe.h) on synthetic Remote
// Play streams. A textured scene moves by `speed` pixels per new picture; a new picture gets coding
// noise, a repeat is the picture before it with a little refinement noise (what the PS5 sends for a
// game below 60 fps: never an exact copy, but changing far less than a new picture). Compressed
// sizes follow the same idea. Each scenario counts, after a warm-up, new pictures called repeats
// (lost to frame generation) and repeats called new (fed to frame generation).
//
// Usage: chiaki-duplicateframetest            all scenarios, exit code 0 when every limit holds

#include "duplicateframe.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct Scene
{
    int w = 640, h = 360, bytes = 1;
    double max_value = 255.0;
    std::vector<float> texture; // 2w x h, scrolled horizontally
    std::mt19937 rng{ 12345 };

    void init(int width, int height, int sample_bytes, double full_scale)
    {
        w = width;
        h = height;
        bytes = sample_bytes;
        max_value = full_scale;
        texture.assign(static_cast<size_t>(2 * w) * static_cast<size_t>(h), 0.0f);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<float> raw(texture.size());
        for (auto &v : raw)
            v = u(rng);
        // A smoothed noise texture with some structure (blocks of different brightness).
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < 2 * w; x++) {
                float s = 0.0f;
                int n = 0;
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        const int yy = std::min(h - 1, std::max(0, y + dy));
                        const int xx = std::min(2 * w - 1, std::max(0, x + dx));
                        s += raw[static_cast<size_t>(yy) * static_cast<size_t>(2 * w) + static_cast<size_t>(xx)];
                        n++;
                    }
                }
                const float block = ((x / 48 + y / 40) % 3) * 0.15f;
                texture[static_cast<size_t>(y) * static_cast<size_t>(2 * w) + static_cast<size_t>(x)] =
                    0.15f + block + 0.4f * (s / n);
            }
        }
    }

    // The picture at scroll position `pos` (pixels) with Gaussian noise of `noise` codes (8-bit).
    std::vector<uint8_t> render(double pos, double noise, double brightness = 0.0)
    {
        std::normal_distribution<double> g(0.0, noise > 0.0 ? noise : 1.0);
        std::vector<uint8_t> out(static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(bytes));
        const int ip = static_cast<int>(std::floor(pos));
        const double fp = pos - ip;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const int x0 = (x + ip) % (2 * w);
                const int x1 = (x0 + 1) % (2 * w);
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(2 * w);
                double v = (1.0 - fp) * texture[row + static_cast<size_t>(x0)] + fp * texture[row + static_cast<size_t>(x1)];
                v = std::min(1.0, std::max(0.0, v + brightness));
                double code = v * 255.0 + (noise > 0.0 ? g(rng) : 0.0);
                code = std::min(255.0, std::max(0.0, code));
                const size_t i = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
                if (bytes == 1) {
                    out[i] = static_cast<uint8_t>(std::lround(code));
                } else {
                    const uint32_t s = static_cast<uint32_t>(std::lround(code / 255.0 * max_value));
                    out[2 * i] = static_cast<uint8_t>(s & 0xff);
                    out[2 * i + 1] = static_cast<uint8_t>(s >> 8);
                }
            }
        }
        return out;
    }

    // A repeat: the picture `base` with refinement noise of `noise` codes added.
    std::vector<uint8_t> refine(const std::vector<uint8_t> &base, double noise)
    {
        std::normal_distribution<double> g(0.0, noise);
        std::vector<uint8_t> out = base;
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        for (size_t i = 0; i < n; i++) {
            if (bytes == 1) {
                out[i] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, out[i] + g(rng))));
            } else {
                const double s = (out[2 * i] | (out[2 * i + 1] << 8)) + g(rng) * max_value / 255.0;
                const uint32_t v = static_cast<uint32_t>(std::min(max_value, std::max(0.0, s)));
                out[2 * i] = static_cast<uint8_t>(v & 0xff);
                out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
            }
        }
        return out;
    }

    DuplicateFrameDetector::Luma luma(const std::vector<uint8_t> &frame) const
    {
        DuplicateFrameDetector::Luma l;
        l.data = frame.data();
        l.linesize = w * bytes;
        l.width = w;
        l.height = h;
        l.bytes = bytes;
        l.max_value = max_value;
        return l;
    }
};

struct Tally
{
    int frames = 0, news = 0, repeats = 0;
    int lost = 0;   // new pictures called repeats
    int leaked = 0; // repeats called new
    int resets = 0;
    double ms = 0.0;
};

// pattern(i): true for a new picture at stream frame i. speed(i): pixels per new picture.
Tally run(const char *name, Scene &scene, int frames, int warmup, const std::function<bool(int)> &pattern,
          const std::function<double(int)> &speed, const std::function<bool(int)> &damaged = nullptr,
          const std::function<double(int)> &brightness = nullptr, bool print_rules = false)
{
    DuplicateFrameDetector det;
    Tally t;
    double pos = 0.0;
    std::vector<uint8_t> last_new;
    std::mt19937 size_rng(99);
    std::uniform_real_distribution<double> jitter(0.85, 1.15);
    for (int i = 0; i < frames; i++) {
        const bool is_new = i == 0 || pattern(i);
        std::vector<uint8_t> frame;
        size_t size = 0;
        if (is_new) {
            pos += speed(i);
            frame = scene.render(pos, 1.5, brightness ? brightness(i) : 0.0);
            last_new = frame;
            size = static_cast<size_t>((8000.0 + 40000.0 * std::min(1.0, speed(i) / 6.0)) * jitter(size_rng));
        } else {
            frame = scene.refine(last_new, 0.6);
            size = static_cast<size_t>((2000.0 + 8000.0 * std::min(1.0, speed(i) / 6.0)) * jitter(size_rng));
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = det.feed(scene.luma(frame), size, damaged ? damaged(i) : false);
        t.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (print_rules && i >= warmup && i < warmup + 12)
            std::printf("    frame %3d %-6s -> %-6s rule %-9s mean %.5f max %.5f line %.5f gap %d%s\n", i,
                        is_new ? "new" : "repeat", r.repeat ? "repeat" : "new", r.rule, r.mean_diff, r.max_diff, r.line,
                        r.gap, r.reset ? " reset" : "");
        if (r.reset)
            t.resets++;
        if (i < warmup)
            continue;
        t.frames++;
        if (is_new) {
            t.news++;
            if (r.repeat)
                t.lost++;
        } else {
            t.repeats++;
            if (!r.repeat)
                t.leaked++;
        }
    }
    std::printf("%-44s frames %4d new %4d repeats %4d | new lost %3d | repeats leaked %3d | resets %2d | %.3f ms/frame\n",
                name, t.frames, t.news, t.repeats, t.lost, t.leaked, t.resets, t.ms / frames);
    return t;
}

} // namespace

int main(int argc, char **argv)
{
    const bool verbose = argc > 1 && std::strcmp(argv[1], "--verbose") == 0;
    int failures = 0;
    auto check = [&failures](bool ok, const char *what) {
        if (!ok) {
            std::printf("  FAIL: %s\n", what);
            failures++;
        }
    };
    Scene s;
    s.init(640, 360, 1, 255.0);
    const auto speed4 = [](int) { return 4.0; };

    auto t60 = run("60 fps game (every frame new)", s, 300, 30, [](int) { return true; }, speed4, nullptr, nullptr, verbose);
    check(t60.lost == 0, "60 fps: no new picture lost");

    auto t30 = run("30 fps game (new, repeat)", s, 300, 30, [](int i) { return i % 2 == 0; }, speed4, nullptr, nullptr, verbose);
    check(t30.lost == 0, "30 fps: no new picture lost");
    check(t30.leaked <= t30.repeats / 50, "30 fps: at most 2% of repeats leaked");

    auto t20 = run("20 fps game (new, repeat, repeat)", s, 300, 30, [](int i) { return i % 3 == 0; }, speed4);
    check(t20.lost == 0, "20 fps: no new picture lost");
    check(t20.leaked <= t20.repeats / 50, "20 fps: at most 2% of repeats leaked");

    auto t45 = run("45 fps game (new, new, new, repeat)", s, 300, 30, [](int i) { return i % 4 != 3; }, speed4);
    check(t45.lost <= t45.news / 50, "45 fps: at most 2% of new pictures lost");
    check(t45.leaked <= t45.repeats / 10, "45 fps: at most 10% of repeats leaked");

    std::mt19937 mix_rng(7);
    std::vector<bool> mix(400);
    for (size_t i = 0; i < mix.size(); i++)
        mix[i] = i == 0 || !mix[i - 1] || (mix_rng() % 3) != 0; // a repeat never follows a repeat
    auto tmix = run("30-60 fps game (uneven)", s, 400, 30, [&mix](int i) { return static_cast<bool>(mix[static_cast<size_t>(i)]); }, speed4);
    check(tmix.lost <= tmix.news / 50, "uneven: at most 2% of new pictures lost");
    check(tmix.leaked <= tmix.repeats / 10, "uneven: at most 10% of repeats leaked");

    auto tslow = run("60 fps, fast then slow pan (6 -> 0.75 px)", s, 300, 30, [](int) { return true; },
                     [](int i) { return i < 150 ? 6.0 : 0.75; });
    check(tslow.lost <= 8, "deceleration: at most 8 new pictures lost (100 ms) around the change");

    auto tslow30 = run("30 fps, slow pan (1 px per picture)", s, 300, 30, [](int i) { return i % 2 == 0; },
                       [](int) { return 1.0; });
    check(tslow30.lost == 0, "30 fps slow pan: no new picture lost");
    check(tslow30.leaked <= tslow30.repeats / 10, "30 fps slow pan: at most 10% of repeats leaked");

    auto tslow20 = run("20 fps, slow pan (1 px per picture)", s, 300, 30, [](int i) { return i % 3 == 0; },
                       [](int) { return 1.0; });
    check(tslow20.lost == 0, "20 fps slow pan: no new picture lost");
    check(tslow20.leaked <= tslow20.repeats / 5, "20 fps slow pan: at most 20% of repeats leaked");

    auto tslowmix = run("30-60 fps, slow pan (1 px per picture)", s, 400, 30,
                        [&mix](int i) { return static_cast<bool>(mix[static_cast<size_t>(i)]); }, [](int) { return 1.0; });
    check(tslowmix.lost <= tslowmix.news / 50, "uneven slow pan: at most 2% of new pictures lost");

    // A 60 fps game whose encoder now and then sends a new picture at a third of the usual size.
    {
        DuplicateFrameDetector det;
        double pos = 0.0;
        int lost = 0, news = 0;
        for (int i = 0; i < 300; i++) {
            pos += 1.0;
            const auto frame = s.render(pos, 1.5);
            const size_t size = (i % 5 == 3) ? 5000 : 15000;
            const auto r = det.feed(s.luma(frame), size, false);
            if (i >= 30) {
                news++;
                if (r.repeat)
                    lost++;
            }
        }
        std::printf("%-44s frames %4d new %4d repeats %4d | new lost %3d |\n", "60 fps slow pan, some small new frames",
                    news, news, 0, lost);
        check(lost == 0, "60 fps with small new frames: no new picture lost");
    }

    auto tstill = run("still scene (nothing moves)", s, 200, 30, [](int) { return true; }, [](int) { return 0.0; });
    (void)tstill; // nothing moves: either call shows the same picture

    auto tloss = run("30 fps with network losses", s, 300, 30, [](int i) { return i % 2 == 0; }, speed4,
                     [](int i) { return i % 50 == 25; });
    check(tloss.resets >= 5, "losses: the frames after a loss reset interpolation");

    auto tcut = run("30 fps with scene cuts", s, 300, 30, [](int i) { return i % 2 == 0; }, speed4, nullptr,
                    [](int i) { return (i / 80) % 2 ? 0.35 : 0.0; });
    check(tcut.resets >= 2, "cuts: the cut pictures reset interpolation");
    check(tcut.lost == 0, "cuts: no new picture lost");

    Scene hd;
    hd.init(1920, 1080, 2, 65535.0); // P010: 10 bits in the high bits of 16
    auto thd = run("1080p 10-bit, 30 fps", hd, 120, 30, [](int i) { return i % 2 == 0; }, speed4);
    check(thd.lost == 0, "1080p 10-bit: no new picture lost");
    check(thd.leaked <= thd.repeats / 50, "1080p 10-bit: at most 2% of repeats leaked");
    check(thd.ms / 120.0 < 2.0, "1080p 10-bit: under 2 ms a frame");

    std::printf("\nVERDICT: %s\n", failures == 0 ? "ALL SCENARIOS PASSED" : "SOME SCENARIOS FAILED, see above");
    return failures == 0 ? 0 : 1;
}
