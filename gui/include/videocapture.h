// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#ifndef CHIAKI_VIDEOCAPTURE_H
#define CHIAKI_VIDEOCAPTURE_H

// Saves the PS5's video as it arrives (an H.264 / H.265 elementary stream, HDR as sent) for frame
// generation tests: ffmpeg turns it into PNG frames for chiaki-fgdx12test --eval. Ctrl+R while
// streaming (StreamSession::ToggleVideoCapture).
//
// The parameter sets (VPS/SPS/PPS) come in a sample of their own at the start of the stream and
// on a profile switch, so the last ones are kept and written first; writing starts at the next
// keyframe (the caller asks the PS5 for one). The receive thread only hands over copies of the
// samples: a writer thread does the file I/O, so a slow disk cannot hold up the stream.

#include <chiaki/log.h>

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QString>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class VideoCapture
{
public:
    enum State { Idle = 0, WaitingForKeyframe = 1, Writing = 2 };

    ~VideoCapture() { stop(); }

    // Before the stream starts.
    void setCodec(bool hevc, ChiakiLog *log)
    {
        std::lock_guard<std::mutex> lock(mutex);
        this->hevc = hevc;
        this->log = log;
    }

    // Receive thread, every video sample (cheap unless a capture runs: only small samples are scanned).
    void feed(const uint8_t *buf, size_t size)
    {
        if (!buf || !size)
            return;
        const bool small = size <= 4096;
        std::unique_lock<std::mutex> lock(mutex);
        if (state == Idle && !small)
            return;
        const Nals nals = (state != Idle || small) ? scan(buf, size) : Nals();
        if (nals.params && !nals.slice)
            header.assign(buf, buf + size);
        if (state == Idle)
            return;
        if (state == WaitingForKeyframe) {
            // A keyframe, or (if none comes) whatever comes after 3 s: the decoder skips to the next one.
            if (!nals.keyframe && armed.elapsed() < 3000)
                return;
            if (nals.params && !nals.slice)
                return; // the header itself: written below with the first frame
            state = Writing;
            writing.start();
            if (!nals.params && !header.empty())
                pushLocked(header.data(), header.size());
            if (log)
                CHIAKI_LOGI(log, "Video capture: writing from %s", nals.keyframe ? "a keyframe" : "the next frame (no keyframe within 3 s)");
        }
        if (state != Writing)
            return;
        if (writing.elapsed() > static_cast<qint64>(max_seconds) * 1000 || written + queued_bytes + size > kMaxBytes) {
            if (log)
                CHIAKI_LOGI(log, "Video capture: limit reached (%d s or %d MB), stopping", max_seconds,
                            static_cast<int>(kMaxBytes >> 20));
            finishLocked();
            return;
        }
        if (queued_bytes + size > kMaxQueued) {
            // The disk does not keep up: stop rather than grow without bound.
            if (log)
                CHIAKI_LOGW(log, "Video capture: the disk does not keep up, stopping");
            finishLocked();
            return;
        }
        pushLocked(buf, size);
    }

    // GUI thread. False when a capture runs already or the file cannot be created.
    bool arm(const QString &path, int seconds)
    {
        stop();
        QDir().mkpath(QFileInfo(path).absolutePath());
        auto f = std::make_unique<QFile>(path);
        if (!f->open(QIODevice::WriteOnly | QIODevice::Truncate))
            return false;
        std::lock_guard<std::mutex> lock(mutex);
        file = std::move(f);
        file_path = path;
        max_seconds = seconds;
        written = 0;
        queued_bytes = 0;
        queue.clear();
        closing = false;
        state = WaitingForKeyframe;
        armed.start();
        writer = std::thread([this]() { writerMain(); });
        return true;
    }

    // GUI thread (or the destructor): stops, writes what is queued and closes the file.
    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (state != Idle)
                finishLocked();
        }
        if (writer.joinable())
            writer.join();
    }

    State currentState()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return state;
    }

    int seconds()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return state == Writing ? static_cast<int>(writing.elapsed() / 1000) : 0;
    }

    // The last capture's file and bytes written (valid once stop() returned).
    QString path()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return file_path;
    }
    qint64 bytesWritten()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return written;
    }

private:
    static constexpr size_t kMaxBytes = size_t(1) << 30;   // 1 GB
    static constexpr size_t kMaxQueued = size_t(64) << 20; // 64 MB waiting for the disk

    struct Nals
    {
        bool params = false;   // VPS / SPS / PPS
        bool slice = false;    // a coded picture
        bool keyframe = false; // IDR / CRA / BLA (H.265), IDR (H.264)
    };

    Nals scan(const uint8_t *buf, size_t size) const
    {
        Nals n;
        for (size_t i = 0; i + 3 < size; i++) {
            if (buf[i] != 0 || buf[i + 1] != 0 || buf[i + 2] != 1)
                continue;
            const uint8_t b = buf[i + 3];
            if (hevc) {
                const int type = (b >> 1) & 0x3f;
                if (type >= 32 && type <= 34)
                    n.params = true;
                else if (type <= 31)
                    n.slice = true;
                if (type >= 16 && type <= 21)
                    n.keyframe = true;
            } else {
                const int type = b & 0x1f;
                if (type == 7 || type == 8)
                    n.params = true;
                else if (type >= 1 && type <= 5)
                    n.slice = true;
                if (type == 5)
                    n.keyframe = true;
            }
            i += 3;
            if (n.keyframe && n.params)
                break;
        }
        return n;
    }

    void pushLocked(const uint8_t *buf, size_t size)
    {
        queue.emplace_back(buf, buf + size);
        queued_bytes += size;
        cond.notify_one();
    }

    void finishLocked()
    {
        state = Idle;
        closing = true;
        cond.notify_one();
    }

    void writerMain()
    {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            cond.wait(lock, [this]() { return !queue.empty() || closing; });
            while (!queue.empty()) {
                std::vector<uint8_t> chunk = std::move(queue.front());
                queue.pop_front();
                queued_bytes -= chunk.size();
                lock.unlock();
                const qint64 n = file ? file->write(reinterpret_cast<const char *>(chunk.data()), static_cast<qint64>(chunk.size())) : -1;
                lock.lock();
                if (n == static_cast<qint64>(chunk.size()))
                    written += n;
            }
            if (closing)
                break;
        }
        if (file) {
            file->close();
            file.reset();
        }
        closing = false;
    }

    std::mutex mutex;
    std::condition_variable cond;
    std::thread writer;
    std::unique_ptr<QFile> file;
    QString file_path;
    std::deque<std::vector<uint8_t>> queue;
    std::vector<uint8_t> header;
    size_t queued_bytes = 0;
    qint64 written = 0;
    int max_seconds = 30;
    bool hevc = true;
    bool closing = false;
    State state = Idle;
    QElapsedTimer armed;
    QElapsedTimer writing;
    ChiakiLog *log = nullptr;
};

#endif // CHIAKI_VIDEOCAPTURE_H
