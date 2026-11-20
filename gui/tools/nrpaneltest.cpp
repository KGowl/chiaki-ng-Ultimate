// Console test of the DLSS 5 panel (src/qml/NeuralRenderPanel.qml), without a stream: the
// panel is loaded from the application's own QML resources with a stand-in for the Chiaki
// singleton (the settings and window properties the panel uses), driven with key presses
// and mouse clicks, and checked against the stand-in's stored values. Writes a screenshot of
// the panel next to the executable (nrpaneltest.png) and prints PASS / FAIL per step.
//
// Usage: chiaki-nrpaneltest [--shot path.png]

#include <QGuiApplication>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickItem>
#include <QQuickView>
#include <QThread>
#include <QVariantMap>
#include <QtQml>

#include <array>
#include <cmath>
#include <cstdio>

class StandInSettings : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int neuralRenderPasses READ passes WRITE setPasses NOTIFY neuralRenderChanged)
    Q_PROPERTY(bool neuralRenderPerPass READ perPass WRITE setPerPass NOTIFY neuralRenderChanged)
    Q_PROPERTY(int neuralRenderModel READ model WRITE setModel NOTIFY neuralRenderChanged)
    Q_PROPERTY(bool neuralRenderUpscale READ upscale WRITE setUpscale NOTIFY neuralRenderChanged)

public:
    StandInSettings()
    {
        for (auto &t : tunings)
            t = defaults();
    }
    static QVariantMap defaults()
    {
        return { { "style", 0 }, { "intensity", 1.0 }, { "localStructure", 1.0 }, { "localTone", 1.0 },
                 { "autoMask", false }, { "skinStructure", 1.0 } };
    }
    int passes() const { return passes_; }
    void setPasses(int p) { passes_ = std::max(1, std::min(3, p)); emit neuralRenderChanged(); }
    bool perPass() const { return per_pass; }
    void setPerPass(bool v)
    {
        if (v)
            tunings[1] = tunings[2] = tunings[0];
        per_pass = v;
        emit neuralRenderChanged();
        emit neuralRenderTuningChanged();
    }
    int model() const { return model_; }
    void setModel(int m) { model_ = m; emit neuralRenderChanged(); }
    bool upscale() const { return upscale_; }
    void setUpscale(bool v) { upscale_ = v; emit neuralRenderChanged(); }

    Q_INVOKABLE QVariantMap neuralRenderTuning(int pass) const { return tunings[static_cast<size_t>(std::max(0, std::min(2, pass)))]; }
    Q_INVOKABLE QVariant neuralRenderTuningDefault(const QString &key) const { return defaults().value(key); }
    Q_INVOKABLE void setNeuralRenderTuningValue(int pass, const QString &key, const QVariant &value)
    {
        QVariant v = value;
        // As QmlSettings: 0.05 steps, clamped.
        auto clamp = [](double x, double lo, double hi) { return std::max(lo, std::min(hi, std::round(x * 20.0) / 20.0)); };
        if (key == "intensity") v = clamp(value.toDouble(), 0, 1);
        else if (key == "localStructure" || key == "localTone") v = clamp(value.toDouble(), 0, 2);
        else if (key == "skinStructure") v = clamp(value.toDouble(), 0, 2.5);
        else if (key == "style") v = std::max(0, std::min(2, value.toInt()));
        tunings[static_cast<size_t>(pass)][key] = v;
        emit neuralRenderTuningChanged();
    }

    std::array<QVariantMap, 3> tunings;
    int passes_ = 1;
    bool per_pass = false;
    int model_ = 0;
    bool upscale_ = true;

signals:
    void neuralRenderChanged();
    void neuralRenderTuningChanged();
};

class StandInWindow : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString videoEnhancementStatus READ status CONSTANT)
    Q_PROPERTY(bool neuralRenderPaused READ paused WRITE setPaused NOTIFY neuralRenderToggled)
    Q_PROPERTY(bool neuralRenderTunable READ tunable CONSTANT)

public:
    QString status() const { return QStringLiteral("DLSS NR → 2160p 4.0 ms"); }
    bool paused() const { return paused_; }
    void setPaused(bool p) { paused_ = p; emit neuralRenderToggled(!p); }
    bool tunable() const { return true; }
    bool paused_ = false;

signals:
    void neuralRenderToggled(bool enabled);
};

class StandInBackend : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QObject *settings READ settings CONSTANT)
    Q_PROPERTY(QObject *window READ window CONSTANT)

public:
    QObject *settings() { return &settings_; }
    QObject *window() { return &window_; }
    StandInSettings settings_;
    StandInWindow window_;
};

// Counts the panel's closeRequested().
class CloseCounter : public QObject
{
    Q_OBJECT
public slots:
    void hit() { count++; }
public:
    int count = 0;
};

static int g_failed = 0;
static void check(bool ok, const char *what)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (!ok)
        g_failed++;
}

int main(int argc, char **argv)
{
    qputenv("QT_QUICK_CONTROLS_STYLE", "Material");
    QGuiApplication app(argc, argv);
    QString shot = QCoreApplication::applicationDirPath() + "/nrpaneltest.png";
    for (int i = 1; i + 1 < argc; i++) {
        if (QString(argv[i]) == "--shot")
            shot = argv[i + 1];
    }
    StandInBackend backend;
    qmlRegisterSingletonInstance("org.streetpea.chiaking", 1, 0, "Chiaki", &backend);

    QQuickView view;
    view.setResizeMode(QQuickView::SizeViewToRootObject);
    view.setColor(QColor(40, 60, 70));
    view.setSource(QUrl("qrc:/NeuralRenderPanel.qml"));
    if (view.status() != QQuickView::Ready) {
        for (const auto &e : view.errors())
            std::printf("QML error: %s\n", qPrintable(e.toString()));
        std::printf("VERDICT: the panel did not load\n");
        return 2;
    }
    QQuickItem *panel = view.rootObject();
    CloseCounter closes;
    QObject::connect(panel, SIGNAL(closeRequested()), &closes, SLOT(hit()));
    panel->setVisible(true);
    view.show();
    view.requestActivate();
    for (int i = 0; i < 50 && !view.isActive(); i++) {
        QCoreApplication::processEvents();
        QThread::msleep(20);
    }
    panel->forceActiveFocus();

    auto settle = [&] {
        for (int i = 0; i < 5; i++) {
            QCoreApplication::processEvents();
            QCoreApplication::sendPostedEvents();
        }
    };
    auto key = [&](int k, int times = 1) {
        for (int i = 0; i < times; i++) {
            QKeyEvent press(QEvent::KeyPress, k, Qt::NoModifier);
            QKeyEvent release(QEvent::KeyRelease, k, Qt::NoModifier);
            QCoreApplication::sendEvent(&view, &press);
            QCoreApplication::sendEvent(&view, &release);
            settle();
        }
    };
    auto row = [&] { return panel->property("currentRow").toInt(); };
    auto tuning = [&](int pass, const char *k) { return backend.settings_.tunings[static_cast<size_t>(pass)].value(k); };
    auto near = [](double a, double b) { return std::fabs(a - b) < 1e-4; };
    settle();

    check(panel->hasActiveFocus(), "panel has the keyboard focus");
    check(row() == 0, "starts on the first row (neural rendering on/off)");
    key(Qt::Key_Right);
    check(backend.window_.paused_, "Right on row 1: neural rendering off for this run");
    key(Qt::Key_Return);
    check(!backend.window_.paused_, "Return on row 1: on again");
    key(Qt::Key_Down);
    key(Qt::Key_Right);
    check(backend.settings_.model_ == 1, "Model row: CC -> HT");
    key(Qt::Key_Backspace);
    check(backend.settings_.model_ == 0, "Backspace: model back to CC");
    key(Qt::Key_Down); // upscale
    key(Qt::Key_Down); // passes
    check(row() == 3, "Down reaches Passes");
    key(Qt::Key_Down);
    check(row() == 6, "with 1 pass the per-pass rows are skipped (Down goes to Preset)");
    key(Qt::Key_Right);
    check(tuning(0, "style").toInt() == 1, "Preset: Default -> Natural (pass 1)");
    key(Qt::Key_Right);
    check(tuning(0, "style").toInt() == 2, "Preset: Natural -> Cinematic");
    key(Qt::Key_Right);
    check(tuning(0, "style").toInt() == 0, "Preset wraps to Default");
    key(Qt::Key_Down);
    key(Qt::Key_Left, 4);
    check(near(tuning(0, "intensity").toDouble(), 0.8), "Intensity Left x4: 1.00 -> 0.80");
    key(Qt::Key_Right, 10);
    check(near(tuning(0, "intensity").toDouble(), 1.0), "Intensity stops at 1.00");
    key(Qt::Key_Left, 2);
    key(Qt::Key_Backspace);
    check(near(tuning(0, "intensity").toDouble(), 1.0), "Backspace: intensity back to 1.00");
    key(Qt::Key_Down);
    key(Qt::Key_Right, 9);
    check(near(tuning(0, "localStructure").toDouble(), 1.45), "Local structure Right x9: 1.45");
    key(Qt::Key_Down);
    key(Qt::Key_Right, 30);
    check(near(tuning(0, "localTone").toDouble(), 2.0), "Local tone stops at 2.00");
    key(Qt::Key_Down);
    check(row() == 11 - 1, "Down reaches Automatic mask");
    key(Qt::Key_Down);
    check(row() == 0, "Skin structure is skipped while the mask is off (wraps to the top)");
    key(Qt::Key_Up);
    key(Qt::Key_Return);
    check(tuning(0, "autoMask").toBool(), "Return: automatic mask on");
    key(Qt::Key_Down);
    check(row() == 11, "Skin structure reachable with the mask on");
    key(Qt::Key_Right, 2);
    check(near(tuning(0, "skinStructure").toDouble(), 1.1), "Skin structure Right x2: 1.10");
    // Two passes, each with its own controls.
    for (int i = 0; i < 12 && row() != 3; i++)
        key(Qt::Key_Up);
    key(Qt::Key_Right);
    check(backend.settings_.passes_ == 2, "Passes: 1 -> 2");
    key(Qt::Key_Down);
    check(row() == 4, "Customize each pass appears with 2 passes");
    key(Qt::Key_Return);
    check(backend.settings_.per_pass, "Customize each pass on");
    check(near(tuning(1, "localStructure").toDouble(), 1.45), "pass 2 starts from pass 1's controls");
    key(Qt::Key_Down);
    check(row() == 5, "Configure pass row appears");
    key(Qt::Key_Right);
    check(panel->property("tuningPass").toInt() == 1, "Configure pass: pass 2");
    key(Qt::Key_Down, 2); // preset, intensity
    key(Qt::Key_Left, 10);
    check(near(tuning(1, "intensity").toDouble(), 0.5) && near(tuning(0, "intensity").toDouble(), 1.0),
          "intensity of pass 2 only: 0.50 (pass 1 stays 1.00)");
    key(Qt::Key_PageUp);
    check(panel->property("tuningPass").toInt() == 0, "PgUp: back to pass 1");
    check(near(panel->property("tuning").toMap().value("intensity").toDouble(), 1.0), "rows show pass 1's values again");
    // The view does not follow the rows that appeared since it opened: size it to the panel.
    check(panel->implicitHeight() > 400, qPrintable(QString("panel height %1 px with all rows").arg(panel->implicitHeight())));
    view.resize(static_cast<int>(panel->width()), static_cast<int>(panel->implicitHeight()));
    settle();
    const QImage img = view.grabWindow();
    check(!img.isNull() && img.save(shot), qPrintable("screenshot " + shot));
    key(Qt::Key_Escape);
    check(closes.count == 1, "Esc sends closeRequested");
    std::printf("-- %d failed\n", g_failed);
    std::printf(g_failed ? "VERDICT: panel checks FAILED\n" : "VERDICT: panel checks passed\n");
    return g_failed ? 1 : 0;
}

#include "nrpaneltest.moc"
