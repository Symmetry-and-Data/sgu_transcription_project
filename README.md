# SGU Unofficial Transcription Project
This project contains some code for automatically transcribing the SGU podcast into text. Right now, the primary addition is the ASRModel class, which can (hopefully) be used as a plug-in replacement for EncoderDecoderASR from speechbrain. It extends the functionality to allow for processing larger audio files by automatically splitting them up at silent periods.

Note: This project is currently in stasis due to other tools currently being used by the SGU transcripts site. No guarantees that it will continue to work, especially since speechbrain can be a bit finicky with versions

TODO: 
1) Improve performance either via fine tuning or a global model built on top based on writings from skeptics

### Installation:
Note: This project uses speechbrain which doesn't seem to work on windows. I have only tested it on Linux, and python 3.8. The easiest way to install is would be to run pip install -r requirements.txt. Unfortunately, PyTorch can be a bit finicky with cuda, so this may be CPU only. Alternatively you can separately go to https://pytorch.org/get-started/locally/ to get details on how to install the pytorch packages

### Examples:

Basic Usage:
```
python -m asr -s "_assets/SGU884-training-Bob.wav" -o "outputs/SGU884-training-Bob.txt"
```
More advanced:
```
python -m asr -s "_assets/SGU884-training-Bob.wav" -o "outputs/SGU884-training-Bob.txt" --batch_seconds 25 --cpu
```

Note, this was aimed at a GPU with 10gb of VRAM. If you have less, or it's filling up, then you can try reducing the default --batch_seconds option (default=20), eg: 18 `--batch_seconds 18`
