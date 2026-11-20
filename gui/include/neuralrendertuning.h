#ifndef CHIAKI_GUI_NEURALRENDERTUNING_H
#define CHIAKI_GUI_NEURALRENDERTUNING_H

// Live controls of DLSS 5 neural rendering on NVIDIA (NGX feature 18's DLSSNR.* evaluation
// keys, see Dx12Fg::NrTuning), one set per pass. They change the picture from the next frame
// on without re-creating the feature; the pass that changed starts a new temporal history.
struct NeuralRenderTuning
{
    static constexpr int kMaxPasses = 3;

    int style = 0;                // 0 Default, 1 Natural, 2 Cinematic
    float intensity = 1.0f;       // 0..1
    float local_structure = 1.0f; // 0..2
    float local_tone = 1.0f;      // 0..2
    bool auto_mask = false;       // the model finds skin itself
    float skin_structure = 1.0f;  // 0..2.5 (1 = the model's own), only with auto_mask

    bool operator==(const NeuralRenderTuning &o) const
    {
        return style == o.style && intensity == o.intensity && local_structure == o.local_structure
            && local_tone == o.local_tone && auto_mask == o.auto_mask && skin_structure == o.skin_structure;
    }
    bool operator!=(const NeuralRenderTuning &o) const { return !(*this == o); }
};

#endif // CHIAKI_GUI_NEURALRENDERTUNING_H
