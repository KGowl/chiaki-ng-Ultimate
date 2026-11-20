import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Material

import org.streetpea.chiaking

// chiaki-ng's DLSS 5 panel over the video on NVIDIA (End opens and closes it; on AMD End is
// NeuralHub's own panel). Every change applies from the next frame: the controls go with each
// evaluation of NVIDIA's feature 18, the model and the number of passes re-create it (a short
// pause). Keys: Up/Down row, Left/Right change, PgUp/PgDn pass, Backspace default, End/Esc close;
// the mouse works too. While it is open the keys stay away from the console.
Rectangle {
    id: panel

    signal closeRequested()

    readonly property int passes: Chiaki.settings.neuralRenderPasses
    readonly property bool perPass: Chiaki.settings.neuralRenderPerPass && passes > 1
    // The pass whose controls the rows show (pass 1's are every pass's unless each has its own).
    property int editPass: 0
    readonly property int tuningPass: perPass ? Math.min(editPass, passes - 1) : 0
    property var tuning: Chiaki.settings.neuralRenderTuning(tuningPass)
    property int currentRow: 0
    readonly property var rows: [rowEnable, rowModel, rowUpscale, rowPasses, rowPerPass, rowPass, rowPreset,
        rowIntensity, rowStructure, rowTone, rowMask, rowSkin]
    readonly property var presetNames: [qsTr("Default"), qsTr("Natural"), qsTr("Cinematic")]

    width: 600
    implicitHeight: content.implicitHeight + 28
    radius: 10
    color: "#e0121418"
    border.color: "#50ffffff"
    border.width: 1
    focus: visible

    function refresh() {
        tuning = Chiaki.settings.neuralRenderTuning(tuningPass);
    }
    onTuningPassChanged: refresh()
    onVisibleChanged: {
        if (visible) {
            refresh();
            if (!navigable(rows[currentRow]))
                currentRow = 0;
        }
    }

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
        case Qt.Key_PageUp: stepPass(-1); break;
        case Qt.Key_PageDown: stepPass(1); break;
        case Qt.Key_Escape: panel.closeRequested(); break;
        default: return;
        }
        event.accepted = true;
    }

    // One row: label, then a choice (< value >), a slider or a check box, and a reset button.
    component PanelRow: Item {
        id: row
        property string label
        property string valueText
        property int kind: 0 // 0 choice, 1 slider, 2 check
        property real from: 0
        property real to: 1
        property real value: 0
        property bool resettable: kind === 1
        property bool selected: false
        signal activated()
        signal step(int dir)
        signal reset()
        signal setTo(real v)

        Layout.fillWidth: true
        implicitHeight: 38
        opacity: enabled ? 1.0 : 0.45

        Rectangle {
            anchors.fill: parent
            radius: 6
            color: row.selected ? "#33ffffff" : "transparent"
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
                font.pixelSize: 15
                Layout.preferredWidth: 190
                elide: Text.ElideRight
            }
            // choice
            ToolButton {
                visible: row.kind === 0
                text: "‹"
                font.pixelSize: 20
                focusPolicy: Qt.NoFocus
                onClicked: { row.activated(); row.step(-1); }
            }
            Label {
                visible: row.kind === 0
                text: row.valueText
                color: Material.accent
                font.pixelSize: 15
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            ToolButton {
                visible: row.kind === 0
                text: "›"
                font.pixelSize: 20
                focusPolicy: Qt.NoFocus
                onClicked: { row.activated(); row.step(1); }
            }
            // slider
            Slider {
                visible: row.kind === 1
                from: row.from
                to: row.to
                stepSize: 0.05
                snapMode: Slider.SnapAlways
                value: row.value
                focusPolicy: Qt.NoFocus
                Layout.fillWidth: true
                onMoved: { row.activated(); row.setTo(value); }
            }
            // check box
            CheckBox {
                visible: row.kind === 2
                checked: row.value > 0.5
                focusPolicy: Qt.NoFocus
                onClicked: { row.activated(); row.setTo(checked ? 1 : 0); }
            }
            Item {
                visible: row.kind === 2
                Layout.fillWidth: true
            }
            Label {
                visible: row.kind === 1
                text: row.valueText
                color: Material.accent
                font.pixelSize: 15
                horizontalAlignment: Text.AlignRight
                Layout.preferredWidth: 44
            }
            ToolButton {
                text: "↺"
                font.pixelSize: 16
                focusPolicy: Qt.NoFocus
                opacity: row.resettable ? 1 : 0
                enabled: row.resettable
                onClicked: { row.activated(); row.reset(); }
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Default (Backspace)")
            }
        }
    }

    ColumnLayout {
        id: content
        objectName: "panelContent"
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            margins: 14
        }
        spacing: 2

        Label {
            text: qsTr("DLSS 5 neural rendering")
            color: "white"
            font.pixelSize: 19
            font.bold: true
        }
        Label {
            Layout.fillWidth: true
            text: Chiaki.window.videoEnhancementStatus.length ? Chiaki.window.videoEnhancementStatus : qsTr("idle")
            color: "#ffe082"
            font.pixelSize: 13
            elide: Text.ElideRight
        }
        Label {
            Layout.fillWidth: true
            visible: !Chiaki.window.neuralRenderTunable
            text: qsTr("The passes and controls below need NVIDIA's DLSS 5 (nvngx_dlssnr.dll next to chiaki-ng) to be running.")
            color: "#ffab91"
            font.pixelSize: 13
            wrapMode: Text.Wrap
        }
        Item { implicitHeight: 6 }

        PanelRow {
            id: rowEnable
            selected: panel.currentRow === 0
            onActivated: panel.select(rowEnable)
            label: qsTr("Neural rendering")
            valueText: Chiaki.window.neuralRenderPaused ? qsTr("Off") : qsTr("On")
            onStep: Chiaki.window.neuralRenderPaused = !Chiaki.window.neuralRenderPaused
        }
        PanelRow {
            id: rowModel
            selected: panel.currentRow === 1
            onActivated: panel.select(rowModel)
            label: qsTr("Model")
            valueText: Chiaki.settings.neuralRenderModel === 1 ? qsTr("HT (lighter)") : qsTr("CC")
            resettable: Chiaki.settings.neuralRenderModel !== 0
            onStep: Chiaki.settings.neuralRenderModel = Chiaki.settings.neuralRenderModel === 1 ? 0 : 1
            onReset: Chiaki.settings.neuralRenderModel = 0
        }
        PanelRow {
            id: rowUpscale
            selected: panel.currentRow === 2
            onActivated: panel.select(rowUpscale)
            label: qsTr("Upscale to the window")
            valueText: Chiaki.settings.neuralRenderUpscale ? qsTr("On") : qsTr("Off")
            onStep: Chiaki.settings.neuralRenderUpscale = !Chiaki.settings.neuralRenderUpscale
        }
        PanelRow {
            id: rowPasses
            selected: panel.currentRow === 3
            onActivated: panel.select(rowPasses)
            label: qsTr("Passes")
            valueText: panel.passes === 1 ? qsTr("1 pass") : qsTr("%1 passes").arg(panel.passes)
            resettable: panel.passes !== 1
            onStep: (dir) => Chiaki.settings.neuralRenderPasses = Math.max(1, Math.min(3, panel.passes + dir))
            onReset: Chiaki.settings.neuralRenderPasses = 1
        }
        PanelRow {
            id: rowPerPass
            selected: panel.currentRow === 4
            onActivated: panel.select(rowPerPass)
            visible: panel.passes > 1
            label: qsTr("Customize each pass")
            kind: 2
            value: Chiaki.settings.neuralRenderPerPass ? 1 : 0
            onStep: Chiaki.settings.neuralRenderPerPass = !Chiaki.settings.neuralRenderPerPass
            onSetTo: (v) => Chiaki.settings.neuralRenderPerPass = v > 0.5
        }
        PanelRow {
            id: rowPass
            selected: panel.currentRow === 5
            onActivated: panel.select(rowPass)
            visible: panel.perPass
            label: qsTr("Configure pass")
            valueText: qsTr("Pass %1 of %2").arg(panel.tuningPass + 1).arg(panel.passes)
            onStep: (dir) => panel.stepPass(dir)
        }
        PanelRow {
            id: rowPreset
            selected: panel.currentRow === 6
            onActivated: panel.select(rowPreset)
            label: qsTr("Preset")
            valueText: panel.presetNames[panel.tuning.style] || ""
            resettable: panel.tuning.style !== 0
            onStep: (dir) => panel.setValue("style", (panel.tuning.style + dir + 3) % 3)
            onReset: panel.setValue("style", panel.defaultOf("style"))
        }
        PanelRow {
            id: rowIntensity
            selected: panel.currentRow === 7
            onActivated: panel.select(rowIntensity)
            label: qsTr("Intensity")
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
            selected: panel.currentRow === 8
            onActivated: panel.select(rowStructure)
            label: qsTr("Local structure")
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
            selected: panel.currentRow === 9
            onActivated: panel.select(rowTone)
            label: qsTr("Local tone")
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
            selected: panel.currentRow === 10
            onActivated: panel.select(rowMask)
            label: qsTr("Automatic mask")
            kind: 2
            value: panel.tuning.autoMask ? 1 : 0
            onStep: panel.setValue("autoMask", !panel.tuning.autoMask)
            onSetTo: (v) => panel.setValue("autoMask", v > 0.5)
        }
        PanelRow {
            id: rowSkin
            selected: panel.currentRow === 11
            onActivated: panel.select(rowSkin)
            enabled: panel.tuning.autoMask
            label: qsTr("Skin structure")
            kind: 1
            from: 0
            to: 2.5
            value: panel.tuning.skinStructure
            valueText: panel.fmt(value)
            onStep: (dir) => panel.setValue("skinStructure", value + 0.05 * dir)
            onSetTo: (v) => panel.setValue("skinStructure", v)
            onReset: panel.setValue("skinStructure", panel.defaultOf("skinStructure"))
        }

        Item { implicitHeight: 4 }
        Label {
            objectName: "panelFooter"
            Layout.fillWidth: true
            text: panel.perPass
                  ? qsTr("↑↓ select   ←→ change   PgUp/PgDn pass   Backspace default   End close")
                  : qsTr("↑↓ select   ←→ change   Backspace default   End close")
            color: "#b0ffffff"
            font.pixelSize: 12
            horizontalAlignment: Text.AlignHCenter
        }
    }
}
