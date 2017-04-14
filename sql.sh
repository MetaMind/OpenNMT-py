mv data/sql/seq2seq.source data/sql/sql.src
mv data/sql/seq2seq.target data/sql/sql.tgt
head -n 2000  data/sql/sql.src > data/sql/val.src
head -n 2000  data/sql/sql.tgt > data/sql/val.tgt
tail -n +2001 data/sql/sql.src > data/sql/train.src
tail -n +2001 data/sql/sql.tgt > data/sql/train.tgt
for s in src tgt; do for f in data/sql/*.$s; do perl tokenizer.perl -a -no-escape -l en -q  < $f > $f.atok; done; done
python preprocess.py -train_src data/sql/train.src.atok -train_tgt data/sql/train.tgt.atok -valid_src data/sql/val.src.atok -valid_tgt data/sql/val.tgt.atok -save_data data/sql.atok -seq_length 500
python train.py -data data/sql.atok.train.pt -save_model sql.atok.model -gpus 0 -brnn -rnn_size 600 -word_vec_size 300 -start_decay_at 50 -epoch 50 -max_generator_batches 100 -dropout 0.2
python translate.py -gpu 0 -model sql.atok.model_acc_66.35_ppl_13.75_e9.pt -src data/sql/val.src.atok -tgt data/sql/val.tgt.atok -replace_unk -verbose -output sql.atok.val.pred
perl multi-bleu.perl data/sql/val.tgt.atok < sql.atok.val.pred
