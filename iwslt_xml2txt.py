import os
import glob
import subprocess
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

parser = ArgumentParser(description='xml2txt')
parser.add_argument('path')
parser.add_argument('-t', '--tags', nargs='+', default=['seg']) 
parser.add_argument('-th', '--threads', default=8, type=int)
parser.add_argument('-a', '--aggressive', action='store_true')
args = parser.parse_args()

def tokenize(f_txt):
    f_tok = f_txt
    f_tok += '.atok' if args.aggressive else '.tok'
    lang = os.path.splitext(f_txt)[1][1:]
    with open(f_tok, 'w') as fout, open(f_txt) as fin:
        if args.aggressive:
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-a', '-q', '-threads', str(args.threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)
        else:
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-q', '-threads', str(args.threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)

for f_xml in glob.iglob(os.path.join(args.path, '*.xml')):
    print(f_xml)
    f_txt = os.path.splitext(f_xml)[0] 
    with open(f_txt, 'w') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for tag in args.tags:
                for e in doc.findall(tag):
                    fd_txt.write(e.text.strip() + '\n')
    tokenize(f_txt)

xml_tags = ['<url', '<keywords', '<talkid', '<description', '<reviewer', '<translator', '<title', '<speaker']
for f_orig in glob.iglob(os.path.join(args.path, 'train.tags*')):
    print(f_orig)
    f_txt = f_orig.replace('.tags', '')
    with open(f_txt, 'w') as fd_txt, open(f_orig) as fd_orig:
        for l in  fd_orig:
            if not any(tag in l for tag in xml_tags):
                fd_txt.write(l.strip() + '\n')
    tokenize(f_txt)

