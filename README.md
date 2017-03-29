# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

# Requirements

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## WMT'16 Multimodal Translation: Multi30k (de-en)

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
```

### 1) Preprocess the data.

```bash
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; do perl lowercase.perl < $f.atok > $f.atok.low; done; done
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; do perl lowercase.perl < $f.tok > $f.tok.low; done; done
python preprocess.py -train_src data/multi30k/train.en.atok.low -train_tgt data/multi30k/train.de.atok.low -valid_src data/multi30k/val.en.atok.low -valid_tgt data/multi30k/val.de.atok.low -save_data data/multi30k.atok.low -lower
python preprocess.py -train_src data/multi30k/train.en.tok.low -train_tgt data/multi30k/train.de.tok.low -valid_src data/multi30k/val.en.tok.low -valid_tgt data/multi30k/val.de.tok.low -save_data data/multi30k.tok.low -lower
```

### 2) Train the model.

```bash
python train.py -data data/multi30k.atok.low.train.pt -save_model multi30k.atok.low.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2
python train.py -data data/multi30k.tok.low.train.pt -save_model multi30k.tok.low.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model model_name -src data/multi30k/test.en.atok.low -tgt data/multi30k/test.de.atok.low -replace_unk -verbose -output multi30k.atok.low.test.pred
python translate.py -gpu 0 -model model_name -src data/multi30k/test.en.tok.low -tgt data/multi30k/test.de.tok.low -replace_unk -verbose -output multi30k.tok.low.test.pred
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/multi30k/test.de.atok.low < multi30k.atok.low.test.pred
perl multi-bleu.perl data/multi30k/test.de.tok.low < multi30k.tok.low.test.pred
```

## IWSLT'16 (de-en)

### 0) Download the data.

```bash
mkdir -p data/iwslt16
wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz && tar -xf de-en.tgz -C data
```

### 1) Preprocess the data.

```bash
python iwslt_xml2txt.py data/de-en -a
python iwslt_xml2txt.py data/de-en
python preprocess.py -train_src data/de-en/train.de-en.en.atok -train_tgt data/de-en/train.de-en.de.atok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.atok -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.atok -save_data data/iwslt16.atok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009
python preprocess.py -train_src data/de-en/train.de-en.en.tok -train_tgt data/de-en/train.de-en.de.tok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.tok -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.tok -save_data data/iwslt16.tok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009
```

### 2) Train the model.

```bash
python train.py -data data/iwslt16.tok.low.train.pt  -save_model iwslt16.tok.low.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2
python train.py -data data/iwslt16.atok.low.train.pt -save_model iwslt16.atok.low.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model model_name -src data/de-en/IWSLT16.TED.tst2014.de-en.en.atok -tgt data/de-en/IWSLT16.TED.tst2014.de-en.de.atok -replace_unk -verbose -output iwslt.ted.tst2014.de-en.atok.low.pred
python translate.py -gpu 0 -model model_name -src data/de-en/IWSLT16.TED.tst2014.de-en.en.tok -tgt data/de-en/IWSLT16.TED.tst2014.de-en.de.tok -replace_unk -verbose -output iwslt.ted.tst2014.de-en.tok.low.pred
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/de-en/IWSLT16.TED.tst2014.de-en.de.atok < iwslt.ted.tst2014.de-en.atok.low.pred
perl multi-bleu.perl data/de-en/IWSLT16.TED.tst2014.de-en.de.tok < iwslt.ted.tst2014.de-en.tok.low.pred
```
