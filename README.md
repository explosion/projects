<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# Example projects

This repo contains example projects for various NLP tasks, including scripts, benchmarks, results and datasets created with [Prodigy](https://prodi.gy).

## 💝 Projects

| Name                                         | Description                                                                                                                                                                                                                                                        | Best result |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------: |
| [`ner-fashion-brands`](ner-fashion-brands)   | Use [`sense2vec`](https://github.com/explosion/sense2vec) to boostrap an NER model to detect fashion brands in Reddit comments. Includes **1735 annotated examples**, a data visualizer, training and evaluation scripts for spaCy and pretrained tok2vec weights. |    82.1 (F) |
| [`ner-drugs`](ner-drugs)                     | Use word vectors to boostrap an NER model to detect drug names in Reddit comments. Includes **1977 annotated examples**, a data visualizer, training and evaluation scripts for spaCy and pretrained tok2vec weights.                                              |    80.6 (F) |
| [`textcat-docs-issues`](textcat-docs-issues) | Train a binary text classifier with exclusive classes to predict whether a GitHub issue title is about documentation. Includes **1161 annotated examples**, a live demo and downloadable model and training and evaluation scripts for spaCy.                                                          |    91.9 (F) |
