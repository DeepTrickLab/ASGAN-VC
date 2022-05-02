from melgan.modules import Generator, Audio2Mel
import torch


def get_default_device():
    return "cpu"
    # if torch.cuda.is_available():
    # return "cuda"
    # else:


def load_model(mel2wav_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    netG = Generator(80, 32, 3).to(device)
    netG.load_state_dict(torch.load("linda_johnson.pt", map_location=device))
    return netG


class MelVocoder:
    def __init__(
        self, device=get_default_device(), model_name="multi_speaker",
    ):
        self.fft = Audio2Mel().to(device)
        netG = Generator(80, 32, 3).to(device)
        netG.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
        self.mel2wav = netG
        self.device = device

    def __call__(self, audio):
        """
        Performs audio to mel conversion (See Audio2Mel in mel2wav/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: log-mel-spectrogram computed on input audio (batch_size, 80, timesteps)
        """
        return self.fft(audio.unsqueeze(1).to(self.device))

    def inverse(self, mel):
        """
        Performs mel2audio conversion
        Args:
            mel (torch.tensor): PyTorch tensor containing log-mel spectrograms (batch_size, 80, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)

        """
        with torch.no_grad():
            return self.mel2wav(mel.to(self.device)).squeeze(1)
