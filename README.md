# SGU Unofficial Transcription Project
This project contains some code for automatically transcribing the SGU podcast into text. Right now, the primary addition is the ASRModel class, which can (hopefully) be used as a plug-in replacement for EncoderDecoderASR from speechbrain. It extends the functionality to allow for processing larger audio files by automatically splitting them up at silent periods. 

TODO: 
1) Punctuation and capitalization (might need another model here)
2) Improve performance either via fine tuning or a global model built on top based on writings from skeptics


Example usage:
Basic Usage:
```
python -m asr -s "_assets/SGU884-training-Bob.wav" -o "outputs/SGU884-training-Bob.txt"
```
More advanced:
```
python -m asr -s "_assets/SGU884-training-Bob.wav" -o "outputs/SGU884-training-Bob.txt" --batch_seconds 25 --cpu
```
