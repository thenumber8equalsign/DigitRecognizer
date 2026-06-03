# DigitRecognizer
This is my first adventure into the field of AI. It is the common example of recognizing handwritten digits

### Stuff
I'm on ArchLinux, so I needed to use pyenv to use python3.10 because tensorflow-datasets doesn't install with pip on python 3.14 (latest as of writing)

```bash
sudo pacman -S pyenv
pyenv install 3.10 && pyenv local 3.10.20 && pyenv exec pip install tensorflow-datasets tensorflow
```
