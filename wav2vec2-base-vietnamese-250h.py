from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os, zipfile
from transformers.file_utils import cached_path, hf_bucket_url
from datasets import load_dataset
import soundfile as sf
import torch
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import subprocess

import flask
from flask import request

#set device
DEVICE = "cuda"

# load model and tokenizer
cache_dir = './cache/'
processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
lm_file = hf_bucket_url("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
lm_file = cached_path(lm_file,cache_dir=cache_dir)
with zipfile.ZipFile(lm_file, 'r') as zip_ref:
    zip_ref.extractall(cache_dir)
lm_file = cache_dir + 'vi_lm_4grams.bin'

#Load n-gram LM
def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    # convert ctc blank character representation
    vocab_list[tokenizer.pad_token_id] = ""
    # replace special characters
    vocab_list[tokenizer.unk_token_id] = ""
    # vocab_list[tokenizer.bos_token_id] = ""
    # vocab_list[tokenizer.eos_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = BeamSearchDecoderCTC(alphabet,
                                   language_model=LanguageModel(lm_model))
    return decoder

ngram_lm_model = get_decoder_ngram_model(processor.tokenizer, lm_file)


# define function to read in sound file
def map_to_array(batch):
    speech, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch

def speech_to_text(audio_path):
    #read soundfiles
    ds = map_to_array({
        "file": audio_path
    })
    # infer model
    input_values = processor(
        ds["speech"], 
        sampling_rate=ds["sampling_rate"], 
        return_tensors="pt"
    ).input_values.to(DEVICE)
    model.to(DEVICE)
    logits = model(input_values).logits[0]
    
    # decode ctc output with greedy decode
    #pred_ids = torch.argmax(logits, dim=-1)
    #greedy_search_output = processor.decode(pred_ids)

    # decode ctc output with beam search decode
    beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
    
    #print(greedy_search_output)
    #print(beam_search_output)
    return beam_search_output

#init run to speed up later queries
speech_to_text("adapter_run-16k.wav")

#convert .mp3 and .wav audio to correct format for model (16000hz, mono channel, .wav) and save as new file
def convert_audio(audio_path):
    if "-16k.wav" in audio_path:
        return audio_path
    if "mp3" in audio_path:
        new_audio_path = audio_path.replace(".mp3","-16k.wav")
    else:
        new_audio_path = audio_path.replace(".wav","-16k.wav")
    subprocess.call(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", new_audio_path])
    #data, samplerate = soundfile.read(audio_path)
    #soundfile.write(new_audio_path, data, samplerate, subtype='PCM_16')
    return new_audio_path

#---API---
app = flask.Flask("API for Speech To Text")
app.config["DEBUG"] = True

@app.route('/audio_path', methods=['POST', 'GET'])
def updateCurrentCode():
    default_path = "/workspace/ai_intern/ARI_sound/"
    audio_path = ""
    #if request.method == "POST":
    #    audio_path = request.json['audio_path']
    #else:
    audio_path = request.args.get(audio_path)
    audio_path = default_path + audio_path
    new_audio_path = convert_audio(audio_path)
    #print(audio_path)
    text = speech_to_text(new_audio_path)

    response = flask.jsonify(text)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Content-type','application/json; charset=utf-8')
    response.success = True
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8090, debug=True)
    #from waitress import serve
    #serve(app, host='0.0.0.0', port=8073)