import utmos
import torch

"""
UTMOS score, automatic Mean Opinion Score (MOS) prediction system, 
(reimplemented using the 'utmos' python library)
"""

class UTMOSScore:
    """Predicting score for each audio clip (using utmos python package)."""

    def __init__(self, device=None, ckpt_path=None):
        # device and ckpt_path are ignored, kept for compatibility
        self.device = device
        self.model = utmos.Score()  # utmos handles downloading and device selection internally

    def score(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavs: audio waveform to be evaluated.
                If len(wavs) == 1 or 2, model processes a single clip.
                If len(wavs) == 3, model processes batch input.
        Returns:
            Predicted MOS score(s) as a torch.Tensor.
        """
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)  # [1, T]
        elif len(wavs.shape) == 2:
            wavs = wavs.unsqueeze(0)  # [1, 1, T]
        elif len(wavs.shape) == 3:
            pass  # batch input
        else:
            raise ValueError("Dimension of input tensor needs to be <= 3.")

        wavs = wavs.squeeze(1)  # ensure shape [B, T]

        scores = []
        for wav in wavs:
            wav_np = wav.cpu().numpy()
            sample_rate = 16000  # utmos assumes 16kHz
            score = self.model.calculate_wav(wav_np, sample_rate)
            scores.append(score)

        return torch.tensor(scores)
