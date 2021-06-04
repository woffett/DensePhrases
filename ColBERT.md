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

output:
```
[Jun 03, 06:39:23] Saving (the following) metadata to index/wiki7500/metadata.json ..
Namespace(amp=True, bsize=256, checkpoint='/home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn', chunksize=6.0, collection='/home/ubuntu/paragraphs.tsv', dim=128, doc_maxlen=180, experiment='dirty', index_name='wiki7500', index_root='index/', mask_punctuation=True, query_maxlen=32, rank=-1, root='experiments', run='2021-06-03_05.18.47', similarity='cosine')
```

### Index FAISS
```
python -m colbert.index_faiss --index_root index/ --index_name wiki7500 --partitions 32768 --sample 0.3
```

output:
```
[Jun 03, 07:29:16] Done indexing!
[Jun 03, 07:29:16] Writing index to index/wiki7500/ivfpq.32768.faiss ...
[Jun 03, 07:29:19]

Done! All complete (for slice #1 of 1)!
  IndexIVFPQ size 0 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=1 reserveVecs=33554432
```

### Retrieve
```
nohup python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--queries ~/questions_dev.tsv \
--nprobe 32 --partitions 32768 --faiss_depth 1024 \
--index_root index/ --index_name wiki7500 \
--checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn \
--batch \
> "log_retrieve_$(date +"%Y_%m_%d_%I_%M_%p").log" 2>&1&
```

returns
```
/home/ubuntu/ColBERT/experiments/dirty/retrieve.py/2021-06-03_07.42.52/ranking.tsv
```

### Re-rank
```
nohup python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --queries ~/questions_dev.tsv --partitions 32768 --index_root index/ --index_name wiki7500 --checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn --topk /home/ubuntu/ColBERT/experiments/dirty/retrieve.py/2021-06-03_07.42.52/ranking.tsv --batch --log-scores > "log_rerank_$(date +"%Y_%m_%d_%I_%M_%p").log" 2>&1&
```

returns
```
/home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-03_15.54.05/ranking.tsv
```

### Annotate
```
python utility/evaluate/annotate_EM.py --qas ~/DensePhrases/splits_7500/dev.json --collection ~/paragraphs.tsv --ranking /home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-03_15.54.05/ranking.tsv
```

# Using the ColBERT NQ checkpoint
download the checkpoint first `colbert-400000.dnn`

Plan
> Index, Faiss Index, retrieve --batch --retrieve_only, rerank --batch --log-scores

## Index
```
python -m colbert.index --amp --doc_maxlen 256 --mask-punctuation --bsize 256 --checkpoint ~/colbert-400000.dnn --collection ~/paragraphs.tsv --index_name wiki7500colbert --index_root index/
```

## FAISS
```
python -m colbert.index_faiss --index_root index/ --index_name wiki7500colbert --partitions 16384 --sample 0.5
```

## Retrieve
```
nohup python -m colbert.retrieve \
--amp --doc_maxlen 256 --mask-punctuation --bsize 256 \
--queries ~/questions_dev.tsv \
--nprobe 32 --partitions 16384 --faiss_depth 1024 \
--index_root index/ --index_name wiki7500colbert \
--checkpoint ~/colbert-400000.dnn \
--batch --retrieve_only \
> "log_retrieve_$(date +"%Y_%m_%d_%I_%M_%p").log" 2>&1&
```

produces:
```
/home/ubuntu/ColBERT/experiments/dirty/retrieve.py/2021-06-04_00.03.13/unordered.tsv
```

## Re-rank
```
python -m colbert.rerank --amp --doc_maxlen 256 --mask-punctuation \
--queries ~/questions_dev.tsv --partitions 16384 --index_root index/ --index_name wiki7500colbert \
--checkpoint ~/colbert-400000.dnn \
--topk /home/ubuntu/ColBERT/experiments/dirty/retrieve.py/2021-06-04_00.03.13/unordered.tsv --batch --log-scores
```

outputs:
```
/home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.19.08/ranking.tsv
```

## Annotate
```
python utility/evaluate/annotate_EM.py --qas ~/DensePhrases/splits_7500/dev.json --collection ~/paragraphs.tsv --ranking /home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.19.08/ranking.tsv
```

outputs:
```
/home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.19.08/ranking.tsv.annotated
```

```
success": {
    "1": 0.4613333333333333,
    "5": 0.688,
    "10": 0.76,
    "20": 0.7906666666666666,
    "30": 0.8026666666666666,
    "50": 0.8253333333333334,
    "100": 0.848,
    "1000": 0.888,
    "all": 0.888
},
```

Re-rank with old one
```
python -m colbert.rerank --amp --doc_maxlen 256 --mask-punctuation \
--queries ~/questions_dev.tsv --partitions 32768 --index_root index/ --index_name wiki7500 \
--checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn \
--topk /home/ubuntu/ColBERT/experiments/dirty/retrieve.py/2021-06-04_00.03.13/unordered.tsv --batch --log-scores
```

outputs
```
/home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.37.01/ranking.tsv
```

## Annotate
```
python utility/evaluate/annotate_EM.py --qas ~/DensePhrases/splits_7500/dev.json --collection ~/paragraphs.tsv --ranking /home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.37.01/ranking.tsv
```

result is in:
```
/home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.37.01/ranking.tsv.annotated
```

last experiment:
```
nohup python -m colbert.test --amp --doc_maxlen 256 --mask-punctuation --collection ~/paragraphs.tsv --queries ~/questions_dev.tsv \
--checkpoint /home/ubuntu/ColBERT/experiments/dirty/train.py/2021-06-02_23.38.41/checkpoints/colbert.dnn \
--topk /home/ubuntu/ColBERT/experiments/dirty/rerank.py/2021-06-04_01.19.08/ranking.tsv.annotated \
> "log_last_$(date +"%Y_%m_%d_%I_%M_%p").log" 2>&1&
```