## Major Changes

### 1 Text preprocessing is changed.

Dunky11 created a not-so-good way for text preprocessing. Maybe his solutions are appropriate for the previous releases of [NeMo](https://github.com/NVIDIA/NeMo), but now it works stable. I can't find the described scenarios:
*Dunky11 quote*:
> Nemo's text normalizer unfortunately produces a large amount of false positives. For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'. To reduce the amount of false positives we will do a check for unusual symbols or words inside the text and only normalize if necessary.
I checked the described cases and it works fine. Tried to find an issue, but wasn't lucky. Maybe I did it wrong.
I have the code, that wrapped the `NeMo` and added several more preprocessing features, that absolutely required, like char mapping. You can find docs here: [NormalizeText](./preprocess/normalize_text.md)

### 2 The phonemizer process (G2P) and tokenization process are changed.

I tried to build the same tokenization as Dunky11, but failed, because of the vocab. It's not possible to reproduce the same vocab in the same order, and the vocab that I have missesed some `IPA` tokens. Change of the vocab order == lost all the progress that was made from the training steps. So, it makes no sense to build my own tokenization that lost benefits during the training process. I decided to use tokenizations from [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer), maybe Dunky11 didn't find it, I don't understand why he's built his own solution.

Maybe because of the `[SILENCE]` token from here:

```python
for sentence in sentence_tokenizer.tokenize(text):
    symbol_ids = []
    sentence = text_normalizer(sentence)
    for word in word_tokenizer.tokenize(sentence):
        word = word.lower()
        if word.strip() == "":
            continue
        elif word in [".", "?", "!"]:
            symbol_ids.append(self.symbol2id[word])
        elif word in [",", ";"]:
            symbol_ids.append(self.symbol2id["SILENCE"])
```

### 3 Training framework instead of tricky training spaghetti

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) instead of wild hacks.
