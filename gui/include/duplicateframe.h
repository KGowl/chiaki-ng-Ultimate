#ifndef CHIAKI_GUI_DUPLICATEFRAME_H
#define CHIAKI_GUI_DUPLICATEFRAME_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

// (std::max) / (std::min) in parentheses: windows.h, included before this header in some files,
// defines max and min as macros.

// Tells a Remote Play stream frame that repeats the last picture from a new picture.
//
// The PS5 always sends 60 frames a second. A game that renders fewer (30 fps, 20 fps, or a rate
// that moves between them) has some pictures sent again: the encoder spends a few bytes refining
// the same picture, so the decoded repeat is never an exact copy, but it changes ten times or more
// less than a new picture does. Frame generation fed with a repeat interpolates between two equal
// pictures and then jumps, so repeats must not reach it.
//
// Decided from the decoded picture (size alone tells them apart badly: a repeat can be nearly
// half a new frame's size, and every frame of a still scene is tiny):
//  - The luma plane is reduced to 16 x 16 pixel block means and compared with the last new
//    picture (not the frame before: a second repeat would then show almost no change and pull the
//    line below the first one): the mean and the largest block difference.
//  - Two bands: over the last kLearnN frames, when the 80th percentile of the mean differences is
//    over kLearnRatio times the 20th (repeats and new pictures), the line between them is their
//    geometric mean. With no frame above the line for more than kStaleBands frames (longer than
//    any cadence leaves between new pictures) the scene slowed down: the bands are learned again.
//  - Without two bands (a 60 fps game, a still scene): a frame changing less than kRepeatRatio of
//    the last new picture's change is a repeat; after a repeat, one changing less than 2.5 times
//    that repeat (or kChainRatio of the last new picture's change) is one too.
//  - In a run of repeats the change from the last new picture stays where the first put it; once
//    it grows past kChainGrow times that, motion is adding up: a slow new picture, not a repeat.
//  - Slow motion (no bands): right after a new picture, a frame that changed less than kTieChange
//    of it and costs under kTieSize of a typical new picture is a repeat.
//  - Rhythm: once the last kRhythmN calls follow a 30 fps (every 2nd frame new) or 20 fps (every
//    3rd) pattern exactly, a call close to the line follows the pattern; a clear exception or
//    losing the bands drops it.
//  - Vetoes: a frame costing at least kSizeVeto of a typical new picture is never a repeat (a
//    repeat costs the encoder far less), and after kMaxGap frames without a new picture the next
//    frame is new anyway. The typical new picture is the 75th percentile of the last kSizeN
//    frames' sizes (whatever they were called, so a wrong call does not move it).
//  - A frame after a network loss (or the PS5's recovery frame), and a scene cut, is new and asks
//    frame generation not to interpolate into it (corruption or two shots would be blended).
// The approach follows SmoovPlay's cadence.DuplicateDetector (learned bands, rhythm, reference to
// the last new frame) and remoteplay-60fps' isDuplicate (relative change, cut test), both MIT; the
// code is our own.
class DuplicateFrameDetector
{
public:
    static constexpr int kBlock = 16; // block side in pixels; every pixel of a block is averaged
    static constexpr int kLearnN = 16;
    static constexpr int kLearnMin = 8;
    static constexpr double kLearnRatio = 4.0;
    static constexpr int kRhythmN = 24;
    static constexpr double kSure = 0.5;
    static constexpr double kRepeatRatio = 0.3;
    static constexpr double kChainRatio = 0.08;
    static constexpr double kIdentical = 0.5 / 255.0; // largest block difference of an exact copy
    static constexpr double kTieChange = 0.6;
    static constexpr double kTieSize = 0.5;
    static constexpr double kSizeVeto = 0.7;
    static constexpr int kSizeN = 16;
    static constexpr int kMaxGap = 6;
    static constexpr double kCutMean = 12.0 / 255.0;
    static constexpr double kCutOverMotion = 2.5;
    static constexpr int kStaleBands = 3;    // frames below the line before the bands count as stale
    static constexpr double kChainGrow = 1.6; // growth of a run's change that marks motion, not a repeat

    // A luma plane in system memory: `bytes` 1 (8-bit) or 2 (16-bit little endian samples);
    // `max_value` the full scale (255, 1023 for low-bit 10-bit, 65535 for high-bit P010).
    struct Luma
    {
        const uint8_t *data = nullptr;
        int linesize = 0;
        int width = 0;
        int height = 0;
        int bytes = 1;
        double max_value = 255.0;
    };

    struct Result
    {
        bool repeat = false;    // the same picture as the last new one: drop it
        int gap = 1;            // a new picture: stream frames since the previous new picture
        bool reset = false;     // a new picture not to interpolate into (after a loss, a scene cut)
        bool compared = false;  // there was a last new picture to compare with
        double mean_diff = 0.0; // mean block difference to the last new picture, 0-1 of the full scale
        double max_diff = 0.0;  // largest block difference, 0-1 of the full scale
        double line = 0.0;      // the learned line between the bands, 0 without two bands
        const char *rule = "";  // first, no-luma, damaged, bands, ratio, growing, identical, size-tie,
                                // rhythm, size, max-gap, cut
    };

    Result feed(const Luma &luma, size_t packet_size, bool damaged)
    {
        Result r;
        since_new++;
        noteSize(packet_size);
        if (!usable(luma)) {
            have_ref = false;
            r.rule = "no-luma";
            return takeNew(r, -1.0);
        }
        const int gw = luma.width / kBlock;
        const int gh = luma.height / kBlock;
        if (gw != grid_w || gh != grid_h) {
            // No grid while ref changes size: a failed allocation must not leave the new grid
            // with the old ref.
            grid_w = grid_h = 0;
            have_ref = false;
            ref.assign(static_cast<size_t>(gw) * static_cast<size_t>(gh), 0.0f);
            grid_w = gw;
            grid_h = gh;
        }
        cur.resize(ref.size());
        blockMeans(luma, cur);
        if (!have_ref) {
            ref.swap(cur);
            have_ref = true;
            r.rule = "first";
            r.reset = damaged; // a loss together with a change of size
            return takeNew(r, -1.0);
        }
        double max_diff = 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < cur.size(); i++) {
            const double d = std::fabs(static_cast<double>(cur[i]) - static_cast<double>(ref[i]));
            max_diff = (std::max)(max_diff, d);
            sum += d;
        }
        r.compared = true;
        r.mean_diff = sum / static_cast<double>(cur.size());
        r.max_diff = max_diff;

        if (damaged) {
            // Not a picture to learn from or to interpolate into.
            calls.clear();
            rhythm_period = 0;
            r.rule = "damaged";
            r.reset = true;
            ref.swap(cur);
            return takeNew(r, r.mean_diff);
        }

        // The call from the picture.
        bool is_new = true;
        double sure = -1.0;
        double lo = 0.0, hi = 0.0;
        bool bands = learnedBands(lo, hi);
        means.push_back(r.mean_diff);
        if (static_cast<int>(means.size()) > kLearnN)
            means.pop_front();
        if (bands) {
            r.line = std::sqrt(lo * hi);
            is_new = r.mean_diff > r.line;
            since_high = is_new ? 0 : since_high + 1;
            if (since_high > kStaleBands) {
                // No frame above the line for longer than any cadence leaves between new pictures:
                // the scene slowed down, the bands are stale. Learn them again from here.
                bands = false;
                r.line = 0.0;
                since_high = 0;
                means.clear();
                means.push_back(r.mean_diff);
            } else {
                sure = std::fabs(std::log((std::max)(r.mean_diff, 1e-9) / r.line)) / std::log(hi / r.line);
                r.rule = "bands";
            }
        }
        if (!bands) {
            r.rule = "ratio";
            if (last_new_mean > 0.0) {
                const double limit = repeats_in_row == 0
                    ? kRepeatRatio * last_new_mean
                    : (std::max)(2.5 * last_repeat_mean, kChainRatio * last_new_mean);
                is_new = !(r.mean_diff < limit);
            }
        }
        // A repeat stays the same picture: in a run of repeats the change from the last new picture
        // stays where the first one put it. When it keeps growing, motion is adding up: a slow new
        // picture (the scene slowed down), not a repeat.
        if (!is_new && repeats_in_row >= 1 && r.mean_diff > kChainGrow * first_repeat_mean
            && r.mean_diff > kChainRatio * last_new_mean) {
            is_new = true;
            r.rule = "growing";
        }
        if (r.max_diff < kIdentical) {
            is_new = false;
            r.rule = "identical";
        }
        // Slow motion leaves a new picture changing only a little more than a repeat. Right after a
        // new picture, a frame that changed clearly less and costs well under a typical new picture
        // is the repeat (only then: after a sudden slowdown the new pictures that follow must not
        // chain into repeats).
        if (is_new && !bands && prev_was_new && last_new_mean > 0.0 && r.mean_diff < kTieChange * last_new_mean
            && packet_size > 0 && typical_size > 0.0 && static_cast<double>(packet_size) < kTieSize * typical_size) {
            is_new = false;
            r.rule = "size-tie";
        }
        is_new = keepRhythm(is_new, sure, r);

        // Vetoes: a frame costing nearly as much as a new picture, or too long without one.
        if (!is_new && packet_size > 0 && typical_size > 0.0
            && static_cast<double>(packet_size) >= kSizeVeto * typical_size) {
            is_new = true;
            r.rule = "size";
        }
        if (!is_new && since_new > kMaxGap) {
            is_new = true;
            r.rule = "max-gap";
        }

        if (!is_new) {
            r.repeat = true;
            r.gap = 0;
            last_repeat_mean = r.mean_diff;
            if (repeats_in_row == 0)
                first_repeat_mean = r.mean_diff;
            repeats_in_row++;
            prev_was_new = false;
            return r;
        }
        if (r.mean_diff > kCutMean && motion_level > 0.0 && r.mean_diff > kCutOverMotion * motion_level) {
            r.reset = true;
            r.rule = "cut";
        }
        ref.swap(cur);
        return takeNew(r, r.mean_diff);
    }

    void reset() { *this = DuplicateFrameDetector(); }

private:
    static bool usable(const Luma &l)
    {
        return l.data && l.width >= kBlock * 4 && l.height >= kBlock * 4 && (l.bytes == 1 || l.bytes == 2)
            && l.max_value >= 1.0 && std::isfinite(l.max_value)
            && static_cast<int64_t>(l.linesize) >= static_cast<int64_t>(l.width) * l.bytes;
    }

    void blockMeans(const Luma &l, std::vector<float> &out) const
    {
        const double scale = 1.0 / (kBlock * kBlock * l.max_value);
        for (int by = 0; by < grid_h; by++) {
            for (int bx = 0; bx < grid_w; bx++) {
                uint64_t sum = 0;
                for (int y = by * kBlock; y < (by + 1) * kBlock; y++) {
                    const uint8_t *row = l.data + static_cast<size_t>(y) * static_cast<size_t>(l.linesize);
                    if (l.bytes == 1) {
                        const uint8_t *p = row + static_cast<size_t>(bx) * kBlock;
                        for (int x = 0; x < kBlock; x++)
                            sum += p[x];
                    } else {
                        const uint8_t *p = row + static_cast<size_t>(bx) * kBlock * 2;
                        for (int x = 0; x < kBlock; x++)
                            sum += static_cast<uint32_t>(p[2 * x]) | (static_cast<uint32_t>(p[2 * x + 1]) << 8);
                    }
                }
                out[static_cast<size_t>(by) * static_cast<size_t>(grid_w) + static_cast<size_t>(bx)] =
                    static_cast<float>(static_cast<double>(sum) * scale);
            }
        }
    }

    // The 20th and 80th percentiles of the recent mean differences, when they form two bands.
    bool learnedBands(double &lo, double &hi) const
    {
        if (static_cast<int>(means.size()) < kLearnMin)
            return false;
        std::vector<double> xs(means.begin(), means.end());
        std::sort(xs.begin(), xs.end());
        const double n = static_cast<double>(xs.size() - 1);
        lo = (std::max)(xs[static_cast<size_t>(std::lround(0.2 * n))], 1e-3 / 255.0);
        hi = xs[static_cast<size_t>(std::lround(0.8 * n))];
        return hi > kLearnRatio * lo;
    }

    // The call after the game's rhythm has had its say, and the rhythm brought up to date.
    bool keepRhythm(bool is_new, double sure, Result &r)
    {
        if (rhythm_period > 0) {
            const bool expected = rhythm_phase == 0;
            if (sure < 0.0 || (expected != is_new && sure >= kSure)) {
                rhythm_period = 0; // no bands, or a clear exception: the pictures alone for a while
                calls.clear();
            } else {
                if (expected != is_new) {
                    is_new = expected;
                    r.rule = "rhythm";
                }
                rhythm_phase = (rhythm_phase + 1) % rhythm_period;
            }
        }
        calls.push_back(is_new);
        if (static_cast<int>(calls.size()) > kRhythmN)
            calls.pop_front();
        if (rhythm_period == 0 && sure >= 0.0 && static_cast<int>(calls.size()) == kRhythmN) {
            int first = -1;
            for (int i = 0; i < kRhythmN; i++) {
                if (calls[static_cast<size_t>(i)]) {
                    first = i;
                    break;
                }
            }
            for (int period : { 2, 3 }) {
                if (first < 0 || first >= period)
                    continue;
                bool steady = true;
                for (int i = 0; i < kRhythmN && steady; i++)
                    steady = calls[static_cast<size_t>(i)] == ((i - first) % period == 0);
                if (steady) {
                    rhythm_period = period;
                    rhythm_phase = (kRhythmN - first) % period;
                    break;
                }
            }
        }
        return is_new;
    }

    // The typical new picture's size: the 75th percentile of the last kSizeN frames' sizes (in a
    // 30 or 20 fps game a third or more of the frames are new, so it lands on a new one).
    void noteSize(size_t packet_size)
    {
        if (!packet_size)
            return;
        sizes.push_back(packet_size);
        if (static_cast<int>(sizes.size()) > kSizeN)
            sizes.pop_front();
        if (sizes.size() < 4) {
            typical_size = 0.0;
            return;
        }
        std::vector<size_t> s(sizes.begin(), sizes.end());
        const size_t k = (s.size() * 3) / 4;
        std::nth_element(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(k), s.end());
        typical_size = static_cast<double>(s[k]);
    }

    Result &takeNew(Result &r, double mean_diff)
    {
        r.repeat = false;
        r.gap = since_new;
        since_new = 0;
        repeats_in_row = 0;
        prev_was_new = true;
        if (mean_diff >= 0.0) {
            last_new_mean = mean_diff;
            motion_level = motion_level > 0.0 ? motion_level + 0.2 * (mean_diff - motion_level) : mean_diff;
        }
        return r;
    }

    int grid_w = 0;
    int grid_h = 0;
    std::vector<float> ref; // the last new picture's block means
    std::vector<float> cur;
    bool have_ref = false;
    std::deque<double> means;
    std::deque<bool> calls;
    int rhythm_period = 0;
    int rhythm_phase = 0;
    double last_new_mean = 0.0;
    double last_repeat_mean = 0.0;
    double first_repeat_mean = 0.0;
    int since_high = 0;
    double motion_level = 0.0;
    std::deque<size_t> sizes;
    double typical_size = 0.0;
    bool prev_was_new = false;
    int repeats_in_row = 0;
    int since_new = 0;
};

#endif // CHIAKI_GUI_DUPLICATEFRAME_H
