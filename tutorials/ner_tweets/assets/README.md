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
{
   "text":"I'm seeing Ed Sheeran on Wednesday in Miami so if you wanna meet up or say hi hmu!",
   "_input_hash":885944384,
   "_task_hash":-1665691389,
   "tokens":[
      {
         "text":"I",
         "start":0,
         "end":1,
         "id":0,
         "ws":false
      },
      {
         "text":"'m",
         "start":1,
         "end":3,
         "id":1,
         "ws":true
      },
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
      {
         "text":"on",
         "start":22,
         "end":24,
         "id":5,
         "ws":true
      },
      {
         "text":"Wednesday",
         "start":25,
         "end":34,
         "id":6,
         "ws":true
      },
      {
         "text":"in",
         "start":35,
         "end":37,
         "id":7,
         "ws":true
      },
      {
         "text":"Miami",
         "start":38,
         "end":43,
         "id":8,
         "ws":true
      },
      {
         "text":"so",
         "start":44,
         "end":46,
         "id":9,
         "ws":true
      },
      {
         "text":"if",
         "start":47,
         "end":49,
         "id":10,
         "ws":true
      },
      {
         "text":"you",
         "start":50,
         "end":53,
         "id":11,
         "ws":true
      },
      {
         "text":"wanna",
         "start":54,
         "end":59,
         "id":12,
         "ws":true
      },
      {
         "text":"meet",
         "start":60,
         "end":64,
         "id":13,
         "ws":true
      },
      {
         "text":"up",
         "start":65,
         "end":67,
         "id":14,
         "ws":true
      },
      {
         "text":"or",
         "start":68,
         "end":70,
         "id":15,
         "ws":true
      },
      {
         "text":"say",
         "start":71,
         "end":74,
         "id":16,
         "ws":true
      },
      {
         "text":"hi",
         "start":75,
         "end":77,
         "id":17,
         "ws":true
      },
      {
         "text":"hmu",
         "start":78,
         "end":81,
         "id":18,
         "ws":false
      },
      {
         "text":"!",
         "start":81,
         "end":82,
         "id":19,
         "ws":true
      }
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
[Prodigy](prodi.gy).  It's a nifty tool that integrates well with spaCy. I don't
have to write any preprocessors on my end, we just have to run 

```sh
prodigy db-out ner_tweets_test > test.jsonl
```

and we're all set!