from speechbrain.pretrained import EncoderDecoderASR
import glob
import torch
import time
from pydub import AudioSegment
import os
import tempfile
from tqdm import tqdm
from pydub.silence import detect_silence
import librosa
import numpy as np
from typing import Optional


def clip(number: float, lower_bond: float, upper_bound: float) -> float:
    if upper_bound < lower_bond:
        raise ValueError(f"Lower bound {lower_bond} must be smaller than upper bound {upper_bound}")
    return min(max(number, lower_bond), upper_bound)


def librosa_to_pydub(y, sr):
    """
    Mostly borrowed from Zabir Al Nazi's answer here:
    https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
    """
    # convert from float to uint16
    y = np.array(y * (1 << 15), dtype=np.int16)
    audio_segment = AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=y.dtype.itemsize,
        channels=1
    )
    return audio_segment


class ASRModel(EncoderDecoderASR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_audio(unsplit_file: str, target_directory: str, min_batch_seconds: int = 10, max_batch_seconds: int = 20,
                    min_silence_len: int = 350, silence_thresh: int = -30, tries: int = 5):
        if tries <= 0:
            raise ValueError(f"Parameter tries ({tries}) should be greater than 0")
        if not os.path.exists(unsplit_file):
            raise OSError(f"File Not Found unsplit_file={unsplit_file}")
        if not os.path.exists(target_directory):
            raise OSError(f"Directory Not Found target_directory={target_directory}")
        # Using librosa because it handles bitrate conversions
        y, s = librosa.load('_assets/SGU884-training-Bob.wav', sr=16000)
        audio_segment = librosa_to_pydub(y, s)

        splitting_successful = False
        split_points = []
        for _ in range(tries):
            if splitting_successful:
                break

            silent_bits = detect_silence(audio_segment,
                                         min_silence_len=min_silence_len,
                                         silence_thresh=silence_thresh
                                         )
            last_split = [0, 0]
            split_points = [0]
            target = max_batch_seconds * 1000 - min_silence_len // 2
            for start_time, end_time in silent_bits:
                if target < start_time:
                    # + min_silence_len // 2 makes it as large as possible while keeping a reasonable amount of silence
                    split_point_candidate = last_split[1] - min_silence_len // 2
                    if split_point_candidate - split_points[-1] < min_batch_seconds * 1000:
                        print(f"No acceptable splits on silence for min_silence_len = {min_silence_len} and "
                              f"silence_thresh={silence_thresh}")
                        min_silence_len = int(min_silence_len * 0.7)
                        silence_thresh = int(silence_thresh * 0.7)
                        print(
                            f"Trying again with min_silence_len = {min_silence_len} and silence_thresh={silence_thresh}")
                        break
                    else:
                        split_points.append(split_point_candidate)
                        target = split_point_candidate + max_batch_seconds * 1000 - min_silence_len // 2
                if start_time <= target <= end_time:
                    # If we're lucky and the target is exactly in a silent spot, just make sure we have enough silence on
                    # either side
                    split_point_candidate = clip(target,
                                                 lower_bond=start_time + min_silence_len // 2,
                                                 upper_bound=end_time - min_silence_len // 2)
                    split_points.append(split_point_candidate)
                    target = split_point_candidate + max_batch_seconds * 1000 - min_silence_len // 2
                else:
                    last_split = [start_time, end_time]

                if target >= len(audio_segment):
                    splitting_successful = True
                    # Note: len(audio_segment) is out of index for the segment, but we'll just be slicing
                    split_points.append(len(audio_segment))
                    break
        else:
            # This bit will happen if we find no working parameters, and the outer for loop finishes without a break
            raise TimeoutError(f"Could not find a working set of parameters for min_silence_len and silence_thresh in"
                               f"{tries} tries. Maybe try making the starting values less conservative")

        if split_points is None:
            raise RuntimeError(f"Somehow the list of split_points {split_points} has length < 2. "
                               f"This should not happen.")

        split_chunk_paths = []

        for i, (split_point_start, split_point_end) in enumerate(zip(split_points, split_points[1:])):
            # Export the audio chunk with new bitrate.
            # print(f"Exporting chunk {i}")
            file_base_name = os.path.splitext(os.path.basename(unsplit_file))[0]
            split_chunk_path = os.path.join(target_directory, f"{file_base_name}_{i}.wav")
            audio_segment[split_point_start: split_point_end].export(
                split_chunk_path,
                # bitrate="16000",
                format="wav"
            )
            split_chunk_paths.append(split_chunk_path)

        return split_chunk_paths, split_points

    def transcribe_file(self, path: str, output_path: Optional[str] = None, **kwargs):
        """Transcribes the given audiofile into a sequence of words. Modified from base to do splitting

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        output_path : str
            Optional parameter for a file to save the transcription to. This will save as it goes, but will delete any
            previous file with the same path

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        if output_path is not None and os.path.exists(output_path):
            os.remove(output_path)

        with tempfile.TemporaryDirectory() as directory:
            # TODO: I think we want to stop this from trying to contact the internet. I think it's producing junk
            chunk_paths, split_points = self.split_audio(path, directory, **kwargs)

            batch_transcriptions = []
            for chunk_path in tqdm(chunk_paths, desc="Transcribing Chunks"):
                waveform = self.load_audio(chunk_path)
                # Fake a batch:
                batch = waveform.unsqueeze(0)
                rel_length = torch.tensor([1.0])
                predicted_words, predicted_tokens = self.transcribe_batch(
                    batch, rel_length
                )
                batch_transcriptions.append(predicted_words[0])
                if output_path is not None:
                    with open(output_path, "at") as f:
                        f.write(batch_transcriptions[-1])
                os.remove(chunk_path)

        return " ".join(batch_transcriptions)


def main():
    asr_model = ASRModel.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                      savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                      run_opts={"device": "cuda:0"})

    asr_model.transcribe_file("_assets/SGU884-training-Steve.wav",
                                              output_path="outputs/SGU884-training-Steve.txt",
                                              max_batch_seconds=20)



if __name__ == "__main__":
    raise SystemExit(main())
