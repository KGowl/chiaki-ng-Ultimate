import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.streetpea.chiaking

// chiaki-ng's "RTX settings" panel over the video, laid out like chiaki-ngx's: two tabs
// (RTX settings, Eye Candy), a Close button, and one row per setting with a drop-down, a check
// box or a slider. Opens with the RTX settings button at the top right of the stream (shown
// while the mouse moves) on every GPU, and with End on NVIDIA (on AMD End is NeuralHub's own
// panel, which holds DLSS 5's model and controls there). Every change applies from the next
// frame: the DLSS 5 controls go with each evaluation of NVIDIA's feature 18, the model and the
// number of passes re-create it (a short pause).
// Keys: Up/Down row, Left/Right change, Tab switch tab, PgUp/PgDn pass, Backspace default,
// End/Esc close. While it is open the keys and the mouse stay away from the console.
Rectangle {
    id: panel

    signal closeRequested()

    readonly property bool nvidia: Chiaki.window.nvidiaCard
    readonly property int passes: Chiaki.settings.neuralRenderPasses
    readonly property bool perPass: Chiaki.settings.neuralRenderPerPass && passes > 1
    // The pass whose controls the rows show (pass 1's are every pass's unless each has its own).
    property int editPass: 0
    readonly property int tuningPass: perPass ? Math.min(editPass, passes - 1) : 0
    property var tuning: Chiaki.settings.neuralRenderTuning(tuningPass)
    property int tab: 0
    property int currentRow: 0
    readonly property var rtxRows: [rowEnable, rowModel, rowPasses, rowPerPass, rowPass, rowPreset,
        rowIntensity, rowStructure, rowTone, rowMask, rowSkin, rowUpscale, rowSharpness, rowStats, rowFrameGen]
    readonly property var candyRows: [rowVsr, rowCleanup, rowStatsStyle, rowStatsPosition, rowBorder]
    readonly property var rows: tab === 0 ? rtxRows : candyRows
    readonly property var presetNames: [qsTr("Default"), qsTr("Natural"), qsTr("Cinematic")]
    // NGX VSR.QualityLevel per entry; Artifact Reduction / Streaming need nvngx_vsr.dll 1.8.2+.
    readonly property var vsrModes: Chiaki.settings.rtxVsrExtendedModes ? [0, 1, 2, 3, 4, 8, 9, 10, 11, 21, 23] : [0, 1, 2, 3, 4]
    readonly property var vsrNames: {
        const all = [qsTr("Off"), qsTr("Upscale Low"), qsTr("Upscale Medium"), qsTr("Upscale High"), qsTr("Upscale Ultra"),
            qsTr("Artifact Reduction Low"), qsTr("Artifact Reduction Medium"), qsTr("Artifact Reduction High"),
            qsTr("Artifact Reduction Ultra"), qsTr("Streaming Medium"), qsTr("Streaming Ultra")];
        return all.slice(0, vsrModes.length);
    }
    // The frame generation backend the Frame generation row turns on again (NVIDIA DLSS FG; on
    // AMD it runs through the D3D12 bridge).
    property int lastFrameGenBackend: 1

    width: 620
    implicitHeight: content.implicitHeight + 24
    radius: 10
    color: "#d8121418"
    border.color: "#40ffffff"
    border.width: 1
    focus: visible
    Material.theme: Material.Dark

    function refresh() {
        tuning = Chiaki.settings.neuralRenderTuning(tuningPass);
    }
    onTuningPassChanged: refresh()
    onVisibleChanged: {
        if (visible) {
            refresh();
            if (!navigable(rows[currentRow]))
                currentRow = firstRow();
        }
    }
    // After the rows' visibility has followed the new tab (an AMD card has no RTX VSR row).
    onTabChanged: Qt.callLater(() => { currentRow = firstRow(); })

    Connections {
        target: Chiaki.settings
        function onNeuralRenderTuningChanged() { panel.refresh() }
        function onNeuralRenderChanged() { panel.refresh() }
    }

    function setValue(key, value) {
        Chiaki.settings.setNeuralRenderTuningValue(tuningPass, key, value);
    }
    function defaultOf(key) {
        return Chiaki.settings.neuralRenderTuningDefault(key);
    }
    function navigable(row) {
        return row && row.visible && row.enabled;
    }
    function firstRow() {
        for (let i = 0; i < rows.length; i++) {
            if (navigable(rows[i]))
                return i;
        }
        return 0;
    }
    function move(dir) {
        let i = currentRow;
        for (let n = 0; n < rows.length; n++) {
            i = (i + dir + rows.length) % rows.length;
            if (navigable(rows[i])) {
                currentRow = i;
                return;
            }
        }
    }
    function select(row) {
        const i = rows.indexOf(row);
        if (i >= 0 && navigable(row))
            currentRow = i;
    }
    function stepPass(dir) {
        if (perPass)
            editPass = Math.max(0, Math.min(passes - 1, editPass + dir));
    }
    function fmt(v) {
        return Number(v).toFixed(2);
    }
    function nrEnabled() {
        return Chiaki.settings.neuralRender && !Chiaki.window.neuralRenderPaused;
    }
    function setNrEnabled(on) {
        if (on && !Chiaki.settings.neuralRender)
            Chiaki.settings.neuralRender = true;
        Chiaki.window.neuralRenderPaused = !on;
    }
    // Frame generation row: 0 Off, 1 2x, 2 3x, 3 4x.
    function frameGenIndex() {
        if (Chiaki.settings.frameGenBackend === 0)
            return 0;
        return Math.max(1, Math.min(3, Chiaki.settings.frameGenMultiplier - 1));
    }
    function setFrameGenIndex(i) {
        i = Math.max(0, Math.min(3, i));
        if (i === 0) {
            if (Chiaki.settings.frameGenBackend !== 0)
                lastFrameGenBackend = Chiaki.settings.frameGenBackend;
            Chiaki.settings.frameGenBackend = 0;
            return;
        }
        Chiaki.settings.frameGenMultiplier = i + 1;
        if (Chiaki.settings.frameGenBackend === 0)
            Chiaki.settings.frameGenBackend = lastFrameGenBackend;
    }
    function setStatsShown(on) {
        Chiaki.settings.showStreamStats = on;
    }

    Keys.onPressed: (event) => {
        const row = rows[currentRow];
        switch (event.key) {
        case Qt.Key_Up: move(-1); break;
        case Qt.Key_Down: move(1); break;
        case Qt.Key_Left: if (navigable(row)) row.step(-1); break;
        case Qt.Key_Right: if (navigable(row)) row.step(1); break;
        case Qt.Key_Return:
        case Qt.Key_Enter:
        case Qt.Key_Space:
            if (navigable(row) && row.kind !== 1) row.step(1);
            break;
        case Qt.Key_Backspace:
        case Qt.Key_Delete: if (navigable(row)) row.reset(); break;
        case Qt.Key_Tab:
        case Qt.Key_Backtab: tab = tab === 0 ? 1 : 0; break;
        case Qt.Key_PageUp: stepPass(-1); break;
        case Qt.Key_PageDown: stepPass(1); break;
        case Qt.Key_Escape: panel.closeRequested(); break;
        default: return;
        }
        event.accepted = true;
    }

    // One row: the label, then a drop-down, a slider (with its default button and value) or a
    // check box with its text.
    component PanelRow: Item {
        id: row
        property string label
        property int kind: 0 // 0 drop-down, 1 slider, 2 check box
        // drop-down
        property var model: []
        property int index: 0
        // slider
        property real from: 0
        property real to: 1
        property real stepSize: 0.05
        property real value: 0
        property string valueText
        // check box
        property bool checked: false
        property string checkText
        property bool resettable: kind === 1
        property bool selected: false
        signal activated()
        signal step(int dir)
        signal reset()
        signal setTo(real v)
        signal pick(int i)

        Layout.fillWidth: true
        implicitHeight: 36
        opacity: enabled ? 1.0 : 0.45

        Rectangle {
            anchors.fill: parent
            radius: 6
            color: row.selected ? "#28ffffff" : "transparent"
            border.color: row.selected ? Material.accent : "transparent"
            border.width: 1
        }
        MouseArea {
            anchors.fill: parent
            onPressed: (mouse) => { row.activated(); mouse.accepted = false; }
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 6
            spacing: 8

            Label {
                text: row.label
                color: "white"
                font.pixelSize: 14
                Layout.preferredWidth: 200
                elide: Text.ElideRight
            }
            ComboBox {
                id: combo
                visible: row.kind === 0
                model: row.model
                currentIndex: row.index
                focusPolicy: Qt.NoFocus
                font.pixelSize: 14
                Layout.preferredWidth: 250
                Layout.preferredHeight: 34
                Material.roundedScale: Material.SmallScale
                onActivated: (i) => {
                    row.activated();
                    row.pick(i);
                    currentIndex = Qt.binding(() => row.index);
                }
            }
            Slider {
                visible: row.kind === 1
                from: row.from
                to: row.to
                stepSize: row.stepSize
                snapMode: Slider.SnapAlways
                value: row.value
                focusPolicy: Qt.NoFocus
                Layout.fillWidth: true
                onMoved: { row.activated(); row.setTo(value); }
            }
            ToolButton {
                visible: row.kind === 1
                text: "↺"
                font.pixelSize: 15
                focusPolicy: Qt.NoFocus
                opacity: row.resettable ? 1 : 0.35
                enabled: row.resettable
                onClicked: { row.activated(); row.reset(); }
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Default (Backspace)")
            }
            Label {
                visible: row.kind === 1
                text: row.valueText
                color: "white"
                font.pixelSize: 14
                horizontalAlignment: Text.AlignRight
                Layout.preferredWidth: 46
            }
            CheckBox {
                visible: row.kind === 2
                checked: row.checked
                text: row.checkText
                font.pixelSize: 14
                focusPolicy: Qt.NoFocus
                onClicked: { row.activated(); row.setTo(checked ? 1 : 0); }
            }
            Item {
                visible: row.kind !== 1
                Layout.fillWidth: true
            }
        }
    }

    // A tab of the header (an inline component: it sees no id of this file, the instance binds
    // `current` and handles the click).
    component TabLabel: ToolButton {
        id: tabButton
        property bool current: false
        focusPolicy: Qt.NoFocus
        font.pixelSize: 15
        contentItem: Label {
            text: tabButton.text
            font: tabButton.font
            color: tabButton.current ? Material.accent : "white"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        Rectangle {
            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
                leftMargin: 8
                rightMargin: 8
            }
            height: 2
            color: Material.accent
            visible: tabButton.current
        }
    }

    ColumnLayout {
        id: content
        objectName: "panelContent"
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            margins: 12
        }
        spacing: 1

        RowLayout {
            Layout.fillWidth: true
            spacing: 4
            TabLabel {
                objectName: "tabRtx"
                text: qsTr("RTX settings")
                current: panel.tab === 0
                onClicked: panel.tab = 0
            }
            TabLabel {
                objectName: "tabCandy"
                text: qsTr("Eye Candy")
                current: panel.tab === 1
                onClicked: panel.tab = 1
            }
            Item { Layout.fillWidth: true }
            Button {
                objectName: "closeButton"
                text: qsTr("Close")
                flat: true
                focusPolicy: Qt.NoFocus
                font.pixelSize: 14
                Material.roundedScale: Material.FullScale
                background: Rectangle {
                    implicitWidth: 72
                    implicitHeight: 32
                    radius: 16
                    color: parent.hovered ? "#50ffffff" : "#30ffffff"
                }
                onClicked: panel.closeRequested()
            }
        }
        Label {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            text: {
                const gpu = Chiaki.window.gpuName;
                const status = Chiaki.window.videoEnhancementStatus.length ? Chiaki.window.videoEnhancementStatus : qsTr("idle");
                return gpu.length ? gpu + "  ·  " + status : status;
            }
            color: "#ffe082"
            font.pixelSize: 12
            elide: Text.ElideRight
        }
        Label {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            visible: panel.tab === 0 && !panel.nvidia
            text: qsTr("AMD: DLSS 5's model, passes and controls are in NeuralHub's panel (End).")
            color: "#b0ffffff"
            font.pixelSize: 12
            wrapMode: Text.Wrap
        }
        Label {
            Layout.fillWidth: true
            Layout.leftMargin: 12
            visible: panel.tab === 0 && panel.nvidia && panel.nrEnabled() && !Chiaki.window.neuralRenderTunable
            text: qsTr("The passes and controls below need NVIDIA's DLSS 5 (nvngx_dlssnr.dll next to chiaki-ng) to be running.")
            color: "#ffab91"
            font.pixelSize: 12
            wrapMode: Text.Wrap
        }
        Item { implicitHeight: 4 }

        // ---- RTX settings
        PanelRow {
            id: rowEnable
            visible: panel.tab === 0
            selected: panel.tab === 0 && panel.currentRow === 0
            onActivated: panel.select(rowEnable)
            label: qsTr("Neural Rendering")
            kind: 2
            checkText: qsTr("Enable (DLSS 5)")
            checked: panel.nrEnabled()
            onStep: panel.setNrEnabled(!panel.nrEnabled())
            onSetTo: (v) => panel.setNrEnabled(v > 0.5)
        }
        PanelRow {
            id: rowModel
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 1
            onActivated: panel.select(rowModel)
            label: qsTr("NR model")
            model: [qsTr("CC"), qsTr("HT (lighter)")]
            index: Chiaki.settings.neuralRenderModel === 1 ? 1 : 0
            resettable: Chiaki.settings.neuralRenderModel !== 0
            onStep: Chiaki.settings.neuralRenderModel = Chiaki.settings.neuralRenderModel === 1 ? 0 : 1
            onPick: (i) => Chiaki.settings.neuralRenderModel = i
            onReset: Chiaki.settings.neuralRenderModel = 0
        }
        PanelRow {
            id: rowPasses
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 2
            onActivated: panel.select(rowPasses)
            label: qsTr("NR passes")
            model: [qsTr("1 pass"), qsTr("2 passes"), qsTr("3 passes")]
            index: panel.passes - 1
            resettable: panel.passes !== 1
            onStep: (dir) => Chiaki.settings.neuralRenderPasses = Math.max(1, Math.min(3, panel.passes + dir))
            onPick: (i) => Chiaki.settings.neuralRenderPasses = i + 1
            onReset: Chiaki.settings.neuralRenderPasses = 1
        }
        PanelRow {
            id: rowPerPass
            visible: panel.tab === 0 && panel.nvidia && panel.passes > 1
            selected: panel.tab === 0 && panel.currentRow === 3
            onActivated: panel.select(rowPerPass)
            label: qsTr("Pass settings")
            kind: 2
            checkText: qsTr("Customize each pass")
            checked: Chiaki.settings.neuralRenderPerPass
            onStep: Chiaki.settings.neuralRenderPerPass = !Chiaki.settings.neuralRenderPerPass
            onSetTo: (v) => Chiaki.settings.neuralRenderPerPass = v > 0.5
        }
        PanelRow {
            id: rowPass
            visible: panel.tab === 0 && panel.nvidia && panel.perPass
            selected: panel.tab === 0 && panel.currentRow === 4
            onActivated: panel.select(rowPass)
            label: qsTr("Configure pass")
            model: {
                const names = [];
                for (let i = 0; i < panel.passes; i++)
                    names.push(qsTr("Pass %1").arg(i + 1));
                return names;
            }
            index: panel.tuningPass
            onStep: (dir) => panel.stepPass(dir)
            onPick: (i) => panel.editPass = i
        }
        PanelRow {
            id: rowPreset
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 5
            onActivated: panel.select(rowPreset)
            label: qsTr("NR preset")
            model: panel.presetNames
            index: panel.tuning.style || 0
            resettable: panel.tuning.style !== 0
            onStep: (dir) => panel.setValue("style", (panel.tuning.style + dir + 3) % 3)
            onPick: (i) => panel.setValue("style", i)
            onReset: panel.setValue("style", panel.defaultOf("style"))
        }
        PanelRow {
            id: rowIntensity
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 6
            onActivated: panel.select(rowIntensity)
            label: qsTr("NR Intensity")
            kind: 1
            from: 0
            to: 1
            value: panel.tuning.intensity
            valueText: panel.fmt(value)
            onStep: (dir) => panel.setValue("intensity", value + 0.05 * dir)
            onSetTo: (v) => panel.setValue("intensity", v)
            onReset: panel.setValue("intensity", panel.defaultOf("intensity"))
        }
        PanelRow {
            id: rowStructure
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 7
            onActivated: panel.select(rowStructure)
            label: qsTr("NR Local Structure")
            kind: 1
            from: 0
            to: 2
            value: panel.tuning.localStructure
            valueText: panel.fmt(value)
            onStep: (dir) => panel.setValue("localStructure", value + 0.05 * dir)
            onSetTo: (v) => panel.setValue("localStructure", v)
            onReset: panel.setValue("localStructure", panel.defaultOf("localStructure"))
        }
        PanelRow {
            id: rowTone
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 8
            onActivated: panel.select(rowTone)
            label: qsTr("NR Local Tone")
            kind: 1
            from: 0
            to: 2
            value: panel.tuning.localTone
            valueText: panel.fmt(value)
            onStep: (dir) => panel.setValue("localTone", value + 0.05 * dir)
            onSetTo: (v) => panel.setValue("localTone", v)
            onReset: panel.setValue("localTone", panel.defaultOf("localTone"))
        }
        PanelRow {
            id: rowMask
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 9
            onActivated: panel.select(rowMask)
            label: qsTr("NR Automatic Mask")
            kind: 2
            checkText: qsTr("Enable")
            checked: !!panel.tuning.autoMask
            onStep: panel.setValue("autoMask", !panel.tuning.autoMask)
            onSetTo: (v) => panel.setValue("autoMask", v > 0.5)
        }
        PanelRow {
            id: rowSkin
            visible: panel.tab === 0 && panel.nvidia
            selected: panel.tab === 0 && panel.currentRow === 10
            onActivated: panel.select(rowSkin)
            enabled: !!panel.tuning.autoMask
            label: qsTr("NR Skin Structure")
            kind: 1
            from: 0
            to: 2.5
            value: panel.tuning.skinStructure
            valueText: panel.fmt(value)
            onStep: (dir) => panel.setValue("skinStructure", value + 0.05 * dir)
            onSetTo: (v) => panel.setValue("skinStructure", v)
            onReset: panel.setValue("skinStructure", panel.defaultOf("skinStructure"))
        }
        PanelRow {
            id: rowUpscale
            visible: panel.tab === 0
            selected: panel.tab === 0 && panel.currentRow === 11
            onActivated: panel.select(rowUpscale)
            label: qsTr("DLSS upscaling")
            kind: 2
            checkText: qsTr("Upscale to the window (1080p → 4K)")
            checked: Chiaki.settings.neuralRenderUpscale
            onStep: Chiaki.settings.neuralRenderUpscale = !Chiaki.settings.neuralRenderUpscale
            onSetTo: (v) => Chiaki.settings.neuralRenderUpscale = v > 0.5
        }
        PanelRow {
            id: rowSharpness
            visible: panel.tab === 0
            selected: panel.tab === 0 && panel.currentRow === 12
            onActivated: panel.select(rowSharpness)
            label: qsTr("Sharpness")
            kind: 1
            from: 0
            to: 100
            stepSize: 5
            value: Chiaki.settings.rcasStrength
            valueText: Math.round(value) + "%"
            resettable: Chiaki.settings.rcasStrength !== 0
            onStep: (dir) => Chiaki.settings.rcasStrength = Math.max(0, Math.min(100, Chiaki.settings.rcasStrength + 5 * dir))
            onSetTo: (v) => Chiaki.settings.rcasStrength = Math.round(v)
            onReset: Chiaki.settings.rcasStrength = 0
        }
        PanelRow {
            id: rowStats
            visible: panel.tab === 0
            selected: panel.tab === 0 && panel.currentRow === 13
            onActivated: panel.select(rowStats)
            label: qsTr("Gameplay statistics")
            kind: 2
            checkText: qsTr("Show stats overlay")
            checked: Chiaki.settings.showStreamStats
            onStep: panel.setStatsShown(!Chiaki.settings.showStreamStats)
            onSetTo: (v) => panel.setStatsShown(v > 0.5)
        }
        PanelRow {
            id: rowFrameGen
            visible: panel.tab === 0 && Chiaki.settings.frameGenSupported
            selected: panel.tab === 0 && panel.currentRow === 14
            onActivated: panel.select(rowFrameGen)
            label: qsTr("Frame generation")
            model: [qsTr("Off"), qsTr("2x"), qsTr("3x"), qsTr("4x")]
            index: panel.frameGenIndex()
            onStep: (dir) => panel.setFrameGenIndex(panel.frameGenIndex() + dir)
            onPick: (i) => panel.setFrameGenIndex(i)
        }

        // ---- Eye Candy
        PanelRow {
            id: rowVsr
            // RTX VSR runs on NVIDIA cards only (the build may carry it on any GPU).
            visible: panel.tab === 1 && Chiaki.settings.rtxVsrSupported && panel.nvidia
            selected: panel.tab === 1 && panel.currentRow === 0
            onActivated: panel.select(rowVsr)
            label: qsTr("RTX Video Super Resolution")
            model: panel.vsrNames
            index: Math.max(0, panel.vsrModes.indexOf(Chiaki.settings.rtxVsrQuality))
            onStep: (dir) => Chiaki.settings.rtxVsrQuality = panel.vsrModes[Math.max(0, Math.min(panel.vsrModes.length - 1, index + dir))]
            onPick: (i) => Chiaki.settings.rtxVsrQuality = panel.vsrModes[i]
        }
        PanelRow {
            id: rowCleanup
            visible: panel.tab === 1 && Chiaki.settings.frameGenSupported
            selected: panel.tab === 1 && panel.currentRow === 1
            onActivated: panel.select(rowCleanup)
            label: qsTr("Generated frame clean-up")
            model: [qsTr("Off"), qsTr("Protect HUD and text"), qsTr("HUD, text + keep detail")]
            index: Chiaki.settings.frameGenCleanup
            onStep: (dir) => Chiaki.settings.frameGenCleanup = Math.max(0, Math.min(2, Chiaki.settings.frameGenCleanup + dir))
            onPick: (i) => Chiaki.settings.frameGenCleanup = i
        }
        PanelRow {
            id: rowStatsStyle
            visible: panel.tab === 1
            selected: panel.tab === 1 && panel.currentRow === 2
            onActivated: panel.select(rowStatsStyle)
            label: qsTr("Stats overlay style")
            model: [qsTr("Mini"), qsTr("Compact"), qsTr("Default"), qsTr("Full")]
            index: Chiaki.settings.streamStatsStyle
            onStep: (dir) => Chiaki.settings.streamStatsStyle = Math.max(0, Math.min(3, Chiaki.settings.streamStatsStyle + dir))
            onPick: (i) => Chiaki.settings.streamStatsStyle = i
        }
        PanelRow {
            id: rowStatsPosition
            visible: panel.tab === 1
            selected: panel.tab === 1 && panel.currentRow === 3
            onActivated: panel.select(rowStatsPosition)
            label: qsTr("Stats overlay position")
            model: [qsTr("Top left"), qsTr("Top center"), qsTr("Top right"),
                qsTr("Middle left"), qsTr("Middle center"), qsTr("Middle right"),
                qsTr("Bottom left"), qsTr("Bottom center"), qsTr("Bottom right")]
            index: Chiaki.settings.streamStatsPosition
            onStep: (dir) => Chiaki.settings.streamStatsPosition = Math.max(0, Math.min(8, Chiaki.settings.streamStatsPosition + dir))
            onPick: (i) => Chiaki.settings.streamStatsPosition = i
        }
        PanelRow {
            id: rowBorder
            visible: panel.tab === 1 && Chiaki.settings.ambientBorderSupported
            selected: panel.tab === 1 && panel.currentRow === 4
            onActivated: panel.select(rowBorder)
            label: qsTr("Border fill")
            model: [qsTr("Black"), qsTr("Ambient (blurred picture)")]
            index: Chiaki.settings.ambientBorder ? 1 : 0
            onStep: Chiaki.settings.ambientBorder = !Chiaki.settings.ambientBorder
            onPick: (i) => Chiaki.settings.ambientBorder = i === 1
        }

        Item { implicitHeight: 4 }
        Label {
            objectName: "panelFooter"
            Layout.fillWidth: true
            text: (panel.perPass && panel.tab === 0
                   ? qsTr("↑↓ select   ←→ change   Tab tab   PgUp/PgDn pass   Backspace default   %1 close")
                   : qsTr("↑↓ select   ←→ change   Tab tab   Backspace default   %1 close")).arg(panel.nvidia ? "End" : "Esc")
            color: "#b0ffffff"
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
