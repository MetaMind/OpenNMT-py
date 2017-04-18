#!/usr/bin/env bash
wget -nc https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget -nc https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget -nc https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget -nc https://github.com/moses-smt/mosesdecoder/blob/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
wget -nc https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/mu
