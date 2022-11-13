import speechbrain.decoders
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
from typing import Optional, List, Tuple
import argparse
import datetime


def clip(number: float, lower_bond: float, upper_bound: float) -> float:
    """
    Takes a number, if it is larger than upper_bound, set it to the upper_bound. If it smaller than lower_bound, set it
    to lower_bound. If it's in between, then leave it alone.

    :param number: number to be clipped
    :param lower_bond: Smallest value of output
    :param upper_bound: Largest value of output
    :return:
    """
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
    def split_audio(unsplit_file: str,
                    target_directory: str,
                    min_batch_seconds: int = 10,
                    max_batch_seconds: int = 20,
                    min_silence_len: int = 350,
                    silence_thresh: int = -30,
                    tries: int = 5) -> Tuple[List[str], List[float]]:
        """
        Takes a wave file path as input, and then breaks it down into smaller chunks between min_batch_seconds and
        max_batch_seconds. It will try to split up the file only at places of silence so that it doesn't break a single
        word into two files. Used to extend class to handle larger files which EncoderDecoderASR will just try to
        process all at once

        :param unsplit_file: source file that we want to split into smaller pieces
        :param target_directory: the directory where the split-up files will be saved
        :param min_batch_seconds: a lower bound on the length in seconds of each split-up file
        :param max_batch_seconds: an upper bound on the length in seconds of each split-up file
        :param min_silence_len: the minimum length of a silent segment in ms that the algorithm will split on. If it
            fails to get segments of the right range of lengths, it will adaptively reduce this number.
        :param silence_thresh: the threshold for how loud of sound can be still considered a silent segment. If it
            fails to get segments of the right range of lengths, it will adaptively increase this number.
        :param tries: the number of times that, if the algorithm fails to find a splitting with particular constraints,
            it'll adjust the min_silence_len and silence_thresh and try again.
        :return:
            split_chunk_paths (list): a tuples of paths to the split up wave files
            split_points (list): the places in the original audio where splits happened. Currently unused
        """
        if tries <= 0:
            raise ValueError(f"Parameter tries ({tries}) should be greater than 0")
        if not os.path.exists(unsplit_file):
            raise OSError(f"File Not Found unsplit_file={unsplit_file}")
        if not os.path.exists(target_directory):
            raise OSError(f"Directory Not Found target_directory={target_directory}")
        # Using librosa because it handles bitrate conversions
        y, s = librosa.load(unsplit_file, sr=16000)
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
                        silence_thresh = int(silence_thresh * 1.3)
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

    def transcribe_file(self, path: str,
                        output_path: Optional[str] = None,
                        include_timestamps: bool = False,
                        # punctuate: bool = False,
                        **kwargs) -> str:
        """Transcribes the given audiofile into a sequence of words. Modified from base to do audio file splitting

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        output_path : str
            Optional parameter for a file to save the transcription to. This will save as it goes, but will delete any
            previous file with the same path
        include_timestamps : bool
            whether to add timestamps to the transcript at each break point
        punctuate : bool
            whether to use a large language model to punctuate the raw transcript

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        if output_path is not None and os.path.exists(output_path):
            os.remove(output_path)

        with tempfile.TemporaryDirectory() as directory:
            chunk_paths, split_points = self.split_audio(path, directory, **kwargs)
            batch_transcriptions = []
            for chunk_path in tqdm(chunk_paths, desc="Transcribing Chunks"):
                waveform = self.load_audio(chunk_path, savedir=directory)
                # Fake a batch:
                batch = waveform.unsqueeze(0)
                rel_length = torch.tensor([1.0])
                predicted_words, predicted_tokens = self.transcribe_batch(
                    batch, rel_length
                )
                batch_transcriptions.append(predicted_words[0].lower())
                os.remove(chunk_path)


        if include_timestamps:
            timestamps = [str(datetime.timedelta(seconds=split_point // 1000)) for split_point in split_points[:-1]]
            txt = "\n".join([f"[{timestamp}] {transcription}"
                             for timestamp, transcription in zip(timestamps, batch_transcriptions)])
        else:
            txt = " ".join(batch_transcriptions)

        # if punctuate:
        #     txt = self.punctuate(txt)

        if output_path is not None:
            with open(output_path, "wt") as f:
                f.write(txt)
        return txt

    @staticmethod
    def punctuate(raw_transcript: str):
        from punctuation import RestorePuncts
        return RestorePuncts().punctuate(raw_transcript)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='SGU Automatic Speech Recognition',
        description='Wrapper for speech brain pretrained models. Takes an audio file, breaks it down into chunks at '
                    'silence points, transcribes each audio piece, then combines them and saves to .txt file')

    parser.add_argument('--source_path', '-s',
                        type=str,
                        default="_assets/SGU884-training-Bob.wav",
                        help='path to source file to be transcribed (should be .wav)')

    parser.add_argument('--output_path', '-o',
                        type=str,
                        default=None,
                        help='path to output file (should be .txt)')

    parser.add_argument('--batch_seconds',
                        type=int,
                        default=20,
                        help="maximum number of seconds in a batch. Choice may depend on your VRAM. default=20")

    parser.add_argument('--cpu',
                        action='store_true',
                        default=False,
                        help="Flag to force a run on the CPU instead of CUDA. Very slow. If false, will check to see"
                             "if a cuda device is available, and use it if so. default=False")

    parser.add_argument('--timestamps',
                        action='store_true',
                        default=False,
                        help="Determines whether to save timestamps in the transcription, based on time in the loaded "
                             "file default=False"
                        )

    # parser.add_argument('--punctuate',
    #                     action='store_true',
    #                     default=False,
    #                     help="Determines whether to use a large language model to punctuate the raw transcription. "
    #                          "default=False."
    #                          "\n"
    #                          "Warning, there may be some compatibility issues with speechbrain and cuda, so I did some"
    #                          "sketchy stuff to prevent other parts from breaking if packages are satisfied."
    #                          " "
    #                     )

    return parser


def main(argv=None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    asr_model = ASRModel.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                      savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                      run_opts={"device": device})

    print(asr_model.transcribe_file(args.source_path,
                                    output_path=args.output_path,
                                    max_batch_seconds=args.batch_seconds,
                                    include_timestamps=args.timestamps,
                                    # punctuate=args.punctuate
                                    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
