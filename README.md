# OpenNMT: Open-Source Neural Machine Translation

This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system.

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

# Requirements

=======
## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the Moses BLEU script for evaluation.

```bash
```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

Stanford CoreNLP can also be used to tokenize on the source side, but we do not have a detokenizer for generated sequences based on CoreNLP tokenization. If you would like to setup CoreNLP for English, open up a new tmux session (tmux new -s corenlp) and run the following:

```bash
mkdir corenlp
cd corenlp
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
unzip stanford-corenlp-full-2016-10-31.zip
cd stanford-corenlp-full-2016-10-31
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# exit your tmux session
```

see https://stanfordnlp.github.io/CoreNLP/corenlp-server.html for more information.

## WMT'16 Multimodal Translation: Multi30k (de-en)

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the data.

```bash
mkdir -p data/multi30k
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz
for l in en de; do for f in data/multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
```

The last line of the train and validation files is blank, so the last line of the bash above removes the non-compliant lines.

### 1) Preprocess the data.

Moses tokenization without html escaping (add the -a option after -no-escape for aggressive hypen splitting)

```bash
for l in en de; do for f in data/multi30k/*.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
```

or alternatively, you can use CoreNLP tokenization. If detokenization of generated sequences is important, then make sure to use Moses on the target side. We currently do not have a way to reliably detokenize for generated sequences.

```bash
for s in train val test; do python corenlp_tokenize.py -input-fn data/multi30k/$s.en -output-fn data/multi30k/$s.en.corenlp; done
```

Typically, we lowercase this dataset, as the important comparisons are in uncased BLEU:

```bash
for f in data/multi30k/*.tok; do perl lowercase.perl < $f > $f.low; done # if you ran Moses
for f in data/multi30k/*.corenlp; do perl lowercase.perl < $f > $f.low; done # if you ran corenlp
```

If you would like to use the Moses tokenization for source and target, prepare the data for the model as so:

```bash
python preprocess.py -train_src data/multi30k/train.en.tok.low -train_tgt data/multi30k/train.de.tok.low -valid_src data/multi30k/val.en.tok.low -valid_tgt data/multi30k/val.de.tok.low -save_data data/multi30k.tok.low -lower
```

If you would like to use the CoreNLP tokenization for English and Moses for German:

```bash
python preprocess.py -train_src data/multi30k/train.en.corenlp.low -train_tgt data/multi30k/train.de.tok.low -valid_src data/multi30k/val.en.corenlp.low -valid_tgt data/multi30k/val.de.tok.low -save_data data/multi30k.corenlp-tok.low -lower
```

The extra lower option in the line above will ensure that the vocabulary object converts all words to lowercase before lookup.

If you would like to use GloVe vectors for translation, now's the time:

```bash
python get_glove_for_dict.py data/multi30k.tok.low.src.dict
python get_glove_for_dict.py data/multi30k.corenlp-tok.low.src.dict # for CoreNLP users
```

### 2) Train the model.

```bash
# Make sure to remove the pretrained word vector and detach argument if you decided not to use GloVe vectors above.

# Moses
python train.py -data data/multi30k.tok.low.train.pt -save_model multi30k.tok.low.glove.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2 # -pre_word_vecs_enc data/multi30k.tok.low.src.dict.glove -detach_embed 100000000

# CoreNLP
python train.py -data data/multi30k.corenlp-tok.low.train.pt -save_model multi30k.corenlp-tok.low.600h.300d.0.2dp.brnn.2l.glove.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 20 -max_generator_batches 100 -dropout 0.2 -pre_word_vecs_enc data/multi30k.corenlp-tok.low.src.dict.glove -detach_embed 100000000
```

### 3) Translate sentences.

For Moses and CoreNLP+Moses respectively:

```bash
python translate.py -gpu 0 -model model_name -src data/multi30k/test.en.tok.low -tgt data/multi30k/test.de.tok.low -replace_unk -verbose -output multi30k.tok.low.test.pred

python translate.py -gpu 0 -model model_name -src data/multi30k/test.en.corenlp.low -tgt data/multi30k/test.de.tok.low -replace_unk -verbose -output multi30k.corenlp-tok.low.test.pred
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/multi30k/test.de.tok.low < multi30k.tok.low.test.pred
perl multi-bleu.perl data/multi30k/test.de.corenlp-tok.low < multi30k.corenlp-tok.low.test.pred
```

## IWSLT'16 (de-en)

### 0) Download the data.

```bash
mkdir -p data/iwslt16
wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz && tar -xf de-en.tgz -C data
```

### 1) Preprocess the data.

```bash
python iwslt_xml2txt.py data/de-en
python iwslt_xml2txt.py data/de-en -a
python iwslt_xml2txt.py data/de-en -corenlp

#Moses
python preprocess.py -train_src data/de-en/train.de-en.en.tok -train_tgt data/de-en/train.de-en.de.tok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.tok -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.tok -save_data data/iwslt16.tok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009

#CoreNLP
python preprocess.py -train_src data/de-en/train.de-en.en.corenlp -train_tgt data/de-en/train.de-en.de.tok -valid_src data/de-en/IWSLT16.TED.tst2013.de-en.en.corenlp -valid_tgt data/de-en/IWSLT16.TED.tst2013.de-en.de.tok -save_data data/iwslt16.corenlp-tok.low -lower -src_vocab_size 22822 -tgt_vocab_size 32009

#Glove Vectors
python get_glove_for_dict.py data/iwslt16.tok.low.src.dict
python get_glove_for_dict.py data/iwslt16.corenlp-tok.low.src.dict # for CoreNLP users
```

### 2) Train the model.

```bash
python train.py -data data/iwslt16.tok.low.train.pt  -save_model iwslt16.tok.low.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2 

python train.py -data data/iwslt16.corenlp-tok.low.train.pt  -save_model iwslt16.corenlp-tok.low.300wv.4l.600h.brnn.2dp.glove.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 20 -max_generator_batches 100 -dropout 0.2 -layers 4-detach_embed 100000000 -pre_word_vecs_enc data/iwslt16.corenlp-tok.low.src.dict.glove -batch_size 256
```

### 3) Translate sentences.

```bash
python translate.py -gpu 0 -model model_name -src data/de-en/IWSLT16.TED.tst2014.de-en.en.tok -tgt data/de-en/IWSLT16.TED.tst2014.de-en.de.tok -replace_unk -verbose -output iwslt.ted.tst2014.de-en.tok.low.pred
```

### 4) Evaluate.

```bash
perl multi-bleu.perl data/de-en/IWSLT16.TED.tst2014.de-en.de.tok < iwslt.ted.tst2014.de-en.tok.low.pred
```

##WMt'17 (de-en)

### 0) Download the data.

```bash
mkdir -p data/wmt17
```

### 1) Preporcess the data.

```bash
python wmt_clean.pyt
for l in en de; do for f in data/wmt17/*.clean.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; perl lowercase.perl < $f.tok > $f.tok.low; done; done
for l in en de; do for f in data/wmt17/test/*.$l; do perl tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; perl lowercase.perl < $f.tok > $f.tok.low; done; done
python preprocess.py -train_src data/wmt17/news-commentary-v12.de-en.clean.en.tok.low -train_tgt data/wmt17/news-commentary-v12.de-en.clean.de.tok.low -valid_src data/wmt17/test/newstest2013.en	n.tok.low -valid_tgt data/wmt17/test/newstest2013.de.tok.low -save_data data/news-commentary.tok.low -lower -seq_length 75
python get_glove_for_dict.py data/news-commentary.tok.low.src.dict
```
### 2) Train the model

```bash
python train.py -data data/news-commentary.tok.low.train.pt  -save_model news-commentary.tok.low.fixed_glove.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300  -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2 -pre_word_vecs_enc data/news-commentary.tok.low.src.dict.glove -detach_embed 100000000000 
```

### 3) Translate sentences.

### 4) Evaluate.
