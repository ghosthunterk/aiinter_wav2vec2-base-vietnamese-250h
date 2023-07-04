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



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

List of things you need to do before using the API.
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

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
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

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
