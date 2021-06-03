```
python utility/evaluate/annotate_EM.py --qas ~/DensePhrases/splits_7500/train.json --collection ~/paragraphs.tsv --ranking ~/rankings.tsv
```

produces:
```
/home/ubuntu/rankings.tsv.annotated
```

### triples generation
```
python utility/supervision/triples.py --ranking /home/ubuntu/rankings.tsv.annotated --positives 3,20 --depth 100 --output ../triples
```

### train
```
python -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples ../triples --similarity l2 --collection ~/paragraphs.tsv --queries ~/questions.tsv
```

### Eval
```
python -m colbert.test --amp --doc_maxlen 180 --mask-punctuation --collection ~/paragraphs.tsv --queries ~/questions_dev.tsv --checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn --topk ~/rankings_dev.tsv
```

### Annotate eval
```
python utility/evaluate/annotate_EM.py --qas ~/DensePhrases/splits_7500/dev.json --collection ~/paragraphs.tsv --ranking ~/colbert-rerank.tsv
```

### Index
```
python -m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn --collection ~/paragraphs.tsv --index_name wiki7500 --index_root index/
```

### Index FAISS
```
python -m colbert.index_faiss --index_root index/ --index_name wiki7500 --partitions 32768 --sample 0.3
```

### Re-rank
```
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --queries ~/questions_dev.tsv --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root index/ --index_name wiki7500 --checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn --topk /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/experiments/dirty/test.py/2021-06-03_02.26.52/ranking.tsv --batch --log-scores
```
