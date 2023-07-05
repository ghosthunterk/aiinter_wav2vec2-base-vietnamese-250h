<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://ailab.siu.edu.vn">
    <img src="logo.png" alt="Logo" width="213" height="150">
  </a>

  <h3 align="center">Wav2vec2-Base-Vietnamese-250h</h3>

  <p align="center">
    Vietnamese Speech-To-Text API for ARI SIUBOT
    <br />
    <a href="https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h"><strong>Original Document »</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

Vietnamese Speech-To-Text API module for local use in [SIU AILAB](www.ailab.siu.edu.vn).

Functions:
* Convert Vietnamese's samples of speech to corresponding text content.
* Host and run API locally.

Local use only.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
### Prerequisites

* create environment
  ```sh
  conda create --name wav2vec2_git python=3.9
  conda activate wav2vec2_git
  mkdir wav2vec2_git
  cd wav2vec2_git
  ```
* install correct version of transformers, soundfile datasets and pyctcdecode
  ```sh
  pip install transformers==4.9.2 soundfile datasets==1.11.0 pyctcdecode==v0.1.0
  ```
* install KenLM language model
  ```sh
  pip install https://github.com/kpu/kenlm/archive/master.zip
  ```
* install torch for greedy decode
  ```sh
  pip install torch
  ```
* install flask for API call
  ```sh
  pip install flask
  ```
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ghosthunterk/aiinter_wav2vec2-base-vietnamese-250h.git
   ```
2. Import libraries
   ```sh
   from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
   import os, zipfile
   from transformers.file_utils import cached_path, hf_bucket_url
   from datasets import load_dataset
   import soundfile as sf
   import torch
   import kenlm
   from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
   import subprocess
   #optional, for API calling   
   import flask
   from flask import request
   ```
3. Set device
   ```sh
   DEVICE = "cuda"
   ```
4. Load model and tokenizer
   ```sh
   cache_dir = './cache/'
   processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
   model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
   lm_file = ""
   if not (os.path.exists(cache_dir+'vi_lm_4grams.bin')):
       lm_file = hf_bucket_url("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')
       lm_file = cached_path(lm_file,cache_dir=cache_dir)
       with zipfile.ZipFile(lm_file, 'r') as zip_ref:
           zip_ref.extractall(cache_dir)
   lm_file = cache_dir + 'vi_lm_4grams.bin'
   ```
5. Load n-gram Language Model
   ```sh
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
   ```
6. Define function to read in sound file
   ```sh
   def map_to_array(batch):
       speech, sampling_rate = sf.read(batch["file"])
       batch["speech"] = speech
       batch["sampling_rate"] = sampling_rate
       return batch
   ```
7. Convert .mp3 and .wav audio to correct format for model (16000hz, mono channel, .wav) and save as new file
   ```sh
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
   ```
8. Speech-To-Text function (read in audio file path)
   ```sh
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
   ```
9. Init run to speed up later queries
   ```sh
   speech_to_text("adapter_run-16k.wav")
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Using below code for API call (REST method)
   ```sh
   app = flask.Flask("API for Speech To Text")
   app.config["DEBUG"] = True
   
   @app.route('/audio_path', methods=['POST', 'GET'])
   def updateCurrentCode():
       default_path = ""
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

   #run API on local port 8090 (127.0.0.1:8090)
   if __name__ == '__main__':
       app.run(host="0.0.0.0", port=8090, debug=True)
       #from waitress import serve
       #serve(app, host='0.0.0.0', port=8090)
   ```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Email: [phamgiakiet273@gmail.com](phamgiakiet273@gmail.com)

Project Link: [https://github.com/ghosthunterk/aiinter_wav2vec2-base-vietnamese-250h](https://github.com/ghosthunterk/aiinter_wav2vec2-base-vietnamese-250h)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Vietnamese end-to-end speech recognition using wav2vec 2.0](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
