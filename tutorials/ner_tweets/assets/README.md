# Tweet Dataset from TweetEval

This dataset contains two `.jsonl` files from the
[TweetEval](https://github.com/cardiffnlp/tweeteval) ([EMNLP
2020](https://arxiv.org/pdf/2010.12421.pdf)) dataset. Here, you can see two
important files:
- `train_initial.jsonl`: this will be our initial training dataset and will be
    augmented further. After going through the data augmentation process, it
    will be split further into training and evaluation datasets.
- `test.jsonl`: this will be our held-out test dataset to get our metrics for
    the baseline (i.e., `en_core_web_lg`) and data-augmented models. This was
    produced by manually annotating randomized tweets from TweetEval using
    [Prodigy](https://prodi.gy/).

Let's inspect the files for a bit. The `train_initial.jsonl` dataset only
contains the `text` key, which contains the tweets. Aside from that, there's no
other information to parse. Let's move on to `test.jsonl` and get an example!

(You'll find this same example in the `demo/demo.jsonl` file, and its serialized
version at `demo/demo.spacy`)

```json
# Excerpt of demo/demo.jsonl file
{
   "text":"I'm seeing Ed Sheeran on Wednesday in Miami so if you wanna meet up or say hi hmu!",
   "_input_hash":885944384,
   "_task_hash":-1665691389,
   "tokens":[
   # ... rest of the tokens
      {
         "text":"seeing",
         "start":4,
         "end":10,
         "id":2,
         "ws":true
      },
      {
         "text":"Ed",
         "start":11,
         "end":13,
         "id":3,
         "ws":true
      },
      {
         "text":"Sheeran",
         "start":14,
         "end":21,
         "id":4,
         "ws":true
      },
    # ...rest of the tokens
   ],
   "spans":[
      {
         "token_start":3,
         "token_end":4,
         "start":11,
         "end":21,
         "text":"Ed Sheeran",
         "label":"PERSON",
         "source":"en_core_web_lg",
         "input_hash":885944384
      }
   ],
   "_is_binary":false,
   "_view_id":"ner_manual",
   "answer":"accept",
   "_timestamp":1633999979
}
```


Ok, that may be too hard to parse right off the bat. Notice that we also have a
`text` field, which contains the tweet itself. In addition, we also have the
`tokens` and `spans` fields. The former lists down all the detected tokens,
while the latter shows the annotated entities. In this case, the span `Ed Sheeran` was
annotated as a `PERSON`. A huge chunk of text isn't fun, so let's display it properly:

```python
import spacy
from spacy import displacy
from spacy.tokens import DocBin

db = DocBin()
nlp = spacy.blank("en")
docs = list(db.from_disk("demo/demo.spacy").get_docs(nlp.vocab))  # a single document file
displacy.serve(doc[0], style="ent)
```

![](/assets/demo/demo_screenshot.png)

Lastly, there are other info and metadata present, and these are artifacts from
[Prodigy](https://prodi.gy).  It's a nifty tool that integrates well with spaCy. I don't
have to write any preprocessors on my end, we just have to run 

```sh
prodigy db-out ner_tweets_test > test.jsonl
```

and we're all set!
