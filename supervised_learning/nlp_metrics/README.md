# NLP Metrics

This project implements common Natural Language Processing evaluation metrics:

## Files

| File | Description |
|------|-------------|
| `0-uni_bleu.py` | Unigram BLEU score calculation |
| `1-ngram_bleu.py` | N-gram BLEU score calculation |
| `2-cumulative_bleu.py` | Cumulative N-gram BLEU score calculation |

## Concepts

### BLEU Score
BLEU (Bilingual Evaluation Understudy) measures the quality of machine-translated text by comparing n-gram overlap between a candidate sentence and reference translations, with a brevity penalty for short candidates.

### Brevity Penalty
Penalizes candidate sentences shorter than the closest reference translation: `BP = exp(1 - r/c)` when `c < r`.

### Cumulative BLEU
Combines multiple n-gram scores using a geometric mean with equal weights up to order `n`.
