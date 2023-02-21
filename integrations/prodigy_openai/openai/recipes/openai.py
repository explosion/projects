import abc
import copy
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional
from typing import Tuple, Type, TypeVar

import jinja2
import pydantic
import requests
import srsly
from dotenv import load_dotenv
from prodigy.components import preprocess
from prodigy.util import log, msg, set_hashes
from spacy.language import Language

_ItemT = TypeVar("_ItemT")
_PromptT = TypeVar("_PromptT", bound="PromptExample")

GLOBAL_STYLE = """
.prodigy-container>.prodigy-content {
    white-space: normal;
    border-top: 1px solid #ddd;
}
.cleaned {
    text-align: left;
    font-size: 14px;
}
.cleaned pre {
    background-color: #eee;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    padding: 15px 20px;
    border-radius: 15px;
    white-space: pre-wrap;
}
summary {
    font-weight: bold;
    cursor: pointer;
    font-size: 1.2em;
}
details {
    margin-bottom: 1rem;
}
"""


# OpenAI defaults
class OPENAI_DEFAULTS:
    # endpoints #
    COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/completions"

    # prompt paths #
    # fmt: off
    TEMPLATES_DIR = "openai_templates"
    TEXTCAT_PROMPT_PATH = str(Path(__file__).parent / TEMPLATES_DIR / "textcat_prompt.jinja2")
    NER_PROMPT_PATH = str(Path(__file__).parent / TEMPLATES_DIR / "ner_prompt.jinja2")
    TERMS_PROMPT_PATH = str(Path(__file__).parent / TEMPLATES_DIR / "terms_prompt.jinja2")
    # fmt: on

    # html templates #
    TEXTCAT_HTML_TEMPLATE = """
    <div class="cleaned">
    {{ #label }}
        <centering>
        <h2>OpenAI GPT-3 says: {{ meta.answer }}</h2>
        </centering>
    {{ /label }}
    <details>
        <summary>Show the prompt for OpenAI</summary>
        <pre>{{openai.prompt}}</pre>
    </details>
    <details>
        <summary>Show the response from OpenAI</summary>
        <pre>{{openai.response}}</pre>
    </details>
    </div>
    """
    NER_HTML_TEMPLATE = """
    <div class="cleaned">
    <details>
        <summary>Show the prompt for OpenAI</summary>
        <pre>{{openai.prompt}}</pre>
    </details>
    <details>
        <summary>Show the response from OpenAI</summary>
        <pre>{{openai.response}}</pre>
    </details>
    </div>
    """


class ENV_VARS:
    # Only store the string values in one place, using a class to support
    # syntax like ENV_VARS.HOST
    OPENAI_ORG = "PRODIGY_OPENAI_ORG"
    OPENAI_KEY = "PRODIGY_OPENAI_KEY"


# OpenAI defaults
class OPENAI_DEFAULTS:
    # endpoints #
    COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/completions"

    # prompt paths #
    # fmt: off
    TEMPLATES_DIR = "templates"
    NER_PROMPT_PATH = str(Path(__file__).parent / TEMPLATES_DIR / "ner_prompt.jinja2")
    # fmt: on

    # html templates #
    NER_HTML_TEMPLATE = """
    <div class="cleaned">
    <details>
        <summary>Show the prompt for OpenAI</summary>
        <pre>{{openai.prompt}}</pre>
    </details>
    <details>
        <summary>Show the response from OpenAI</summary>
        <pre>{{openai.response}}</pre>
    </details>
    </div>
    """


@dataclass
class PromptExample(abc.ABC):
    """An example to be passed into an OpenAI prompt.
    When inheriting this dataclass, you should implement the `from_prodigy`
    function that takes in a Prodigy task example and formats it back
    into a dataclass that can fill a prompt template.
    You can refer to Prodigy's API Interfaces documentation
    (https://prodi.gy/docs/api-interfaces) to see how most examples are structured
    for each task.
    """

    @staticmethod
    def is_flagged(example: _ItemT) -> bool:
        """Check whether a Prodigy example is flagged for use
        in the prompt."""

        return (
            example.get("flagged") is True
            and example.get("answer") == "accept"
            and "text" in example
        )

    @classmethod
    def from_prodigy(cls, example: _ItemT, labels: Iterable[str]) -> "PromptExample":
        """Create a prompt example from Prodigy's format."""
        ...


def normalize_label(label: str) -> str:
    return label.lower()


class OpenAISuggester:
    """Suggest annotations using OpenAI's GPT-3
    prompt_template (jinja2.Template): A Jinja2 template that contains the
        prompt to send to OpenAI's GPT-3 API.
    model (str): The GPT-3 model ID to use for completion. Check the OpenAI
        documentation for more information https://beta.openai.com/docs/models/overview.
    labels (List[str]): List of labels for annotation.
    max_examples (int): The maximum number of examples to stream in the Prodigy UI.
    segment (bool): If set to True, segment the documents into sentences.
    verbose (bool): Show verbose output in the command-line, including the prompt and response from OpenAI.
    openai_api_org (str): The OpenAI API organization.
    openai_api_key (str): The OpenAI API key.
    openai_temperature (float): The temperature parameter (from 0 to 1) that controls the
        randomness of GPT-3's output.
    openai_max_tokens (int): The maximum amout of tokens that GPT-3's
        completion API can generate.
    openai_n (int): The number of completions to generate for each prompt.
    openai_n_retries (int): The number of retries whenever a 429 error occurs.
    openai_retry_timeout_s (int): The amount of time before attempting another request whenever we
        encounter a 429 error. Increases exponentially for each retry.
    openai_read_timeout_s (int): The amount of time to wait a response output during a request.
    examples (List[PromptExample]): A list of examples to add to the prompt to guide GPT-3 output.
    response_parser (Callable[str] -> Dict): A function that accepts a string that represents
        GPT-3's raw response, and parses it into a dictionary that is compatible to Prodigy's
        annotation interfaces.
    render_vars (Dict[str, Any]): A dictionary containing additional variables to render in the
        Jinja2 template. By default, the Jinja2 template will render the text (str), some labels (List[str]),
        and examples (PromptExample). If you wish to add other task-specific variables, you should supply
        them in this variable.
    """

    prompt_template: jinja2.Template
    model: str
    labels: List[str]
    max_examples: int
    segment: bool
    verbose: bool
    openai_api_org: str
    openai_api_key: str
    openai_temperature: float
    openai_max_tokens: int
    openai_retry_timeout_s: int
    openai_read_timeout_s: int
    openai_n_retries: int
    openai_n: int
    examples: List[PromptExample]
    response_parser: Callable
    render_vars: Dict[str, Any]
    prompt_example_class: PromptExample

    openai_completions_endpoint: ClassVar[str] = OPENAI_DEFAULTS.COMPLETIONS_ENDPOINT
    retry_error_codes: ClassVar[List[int]] = [429, 503]

    def __init__(
        self,
        prompt_template: jinja2.Template,
        *,
        labels: List[str],
        max_examples: int,
        segment: bool,
        openai_api_org: str,
        openai_api_key: str,
        openai_model: str,
        response_parser: Callable,
        prompt_example_class: PromptExample,
        openai_temperature: int = 0,
        openai_max_tokens: int = 500,
        openai_retry_timeout_s: int = 1,
        openai_read_timeout_s: int = 30,
        openai_n_retries: int = 10,
        openai_n: int = 1,
        render_vars: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.prompt_template = prompt_template
        self.model = openai_model
        self.labels = [normalize_label(label) for label in labels]
        self.max_examples = max_examples
        self.verbose = verbose
        self.segment = segment
        self.examples = []
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_retry_timeout_s = openai_retry_timeout_s
        self.openai_read_timeout_s = openai_read_timeout_s
        self.openai_n = openai_n
        self.openai_n_retries = openai_n_retries
        self.response_parser = response_parser
        self.prompt_example_class = prompt_example_class
        self.render_vars = {} if render_vars is None else render_vars

    def __call__(
        self,
        stream: Iterable[_ItemT],
        *,
        nlp: Language,
        batch_size: int,
        **kwargs,
    ) -> Iterable[_ItemT]:
        if self.segment:
            stream = preprocess.split_sentences(nlp, stream)  # type: ignore

        stream = self.pipe(stream, nlp, batch_size, **kwargs)
        stream = self.set_hashes(stream)
        return stream

    def pipe(
        self, stream: Iterable[_ItemT], nlp: Language, batch_size: int, **kwargs
    ) -> Iterable[_ItemT]:
        """Process the stream and add suggestions from OpenAI."""
        stream = self.stream_suggestions(stream, batch_size)
        stream = self.format_suggestions(stream, nlp=nlp)
        return stream

    def set_hashes(self, stream: Iterable[_ItemT]) -> Iterable[_ItemT]:
        for example in stream:
            yield set_hashes(example)

    def update(self, examples: Iterable[_ItemT]) -> float:
        """Update the examples that will be used in the prompt based on user flags."""
        for eg in examples:
            if PromptExample.is_flagged(eg):
                self.add_example(
                    self.prompt_example_class.from_prodigy(eg, self.labels)
                )
        return 0.0

    def add_example(self, example: PromptExample) -> None:
        """Add an example for use in the prompts. Examples are pruned to the most recent max_examples."""
        if self.max_examples and example:
            self.examples.append(example)
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[-self.max_examples :]

    def stream_suggestions(
        self, stream: Iterable[_ItemT], batch_size: int
    ) -> Iterable[_ItemT]:
        """Get zero-shot or few-shot annotations from OpenAI.
        Given a stream of input examples, we define a prompt, get a response from OpenAI,
        and yield each example with their predictions to the output stream.
        """
        for batch in batch_sequence(stream, batch_size):
            prompts = [
                self._get_prompt(eg["text"], labels=self.labels, examples=self.examples)
                for eg in batch
            ]
            responses = self._get_openai_response(prompts)
            for eg, prompt, response in zip(batch, prompts, responses):
                if self.verbose:
                    log("Prompt to OpenAI", details=prompt)
                eg["openai"] = {"prompt": prompt, "response": response}
                if self.verbose:
                    log("Got response from OpenAI", details=response)
                yield eg

    def format_suggestions(
        self, stream: Iterable[_ItemT], *, nlp: Language
    ) -> Iterable[_ItemT]:
        """Parse the examples in the stream and set up labels
        to display in the Prodigy UI.
        """
        stream = preprocess.add_tokens(nlp, stream, skip=True)  # type: ignore
        for example in stream:
            example = copy.deepcopy(example)
            if "meta" not in example:
                example["meta"] = {}

            response = example["openai"].get("response", "")
            example.update(self.response_parser(response, example))
            yield example

    def _get_prompt(
        self, text: str, labels: List[str], examples: List[PromptExample]
    ) -> str:
        """Generate a prompt for GPT-3 OpenAI."""
        return self.prompt_template.render(
            text=text, labels=labels, examples=examples, **self.render_vars
        )

    def _get_openai_response(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        r = retry(
            lambda: requests.post(
                self.openai_completions_endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": prompts,
                    "temperature": self.openai_temperature,
                    "max_tokens": self.openai_max_tokens,
                    "n": self.openai_n,
                },
                timeout=self.openai_read_timeout_s,
            ),
            n=self.openai_n_retries,
            timeout_s=self.openai_retry_timeout_s,
            error_codes=self.retry_error_codes,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"] for i in range(len(prompts))]


class PromptInput(pydantic.BaseModel):
    id: str
    prompt_args: Dict[str, Any]


class OpenAIPromptAB:
    """Perform A/B testing for prompts
    display (jinja2.Template): A jinja2 template that contains what will be
        displayed in the Prodigy UI.
    prompts (Dict[str, jinja2.Template]): A mapping of prompt names to their corresponding
        prompt template.
    inputs (Iterable[PromptInput]): Input examples to compare prompts.
    batch_size (int): The number of examples to be shown for a given batch.
    verbose (bool): Show verbose output in the command-line, including the
        prompt and response from OpenAI.
    randomize (bool): Randomize the examples when shown in the Prodigy UI.
    openai_api_org (str): The OpenAI API organization.
    openai_api_key (str): The OpenAI API key.
    openai_temperature (float): The temperature parameter (from 0 to 1) that controls the
        randomness of GPT-3's output.
    openai_max_tokens (int): The maximum amout of tokens that GPT-3's
        completion API can generate.
    openai_n (int): The number of completions to generate for each prompt.
    openai_n_retries (int): The number of retries whenever a 429 error occurs.
    openai_retry_timeout_s (int): The amount of time before attempting another request whenever we
        encounter a 429 error. Increases exponentially for each retry.
    openai_read_timeout_s (int): The amount of time to wait a response output during a request.
    """

    display: jinja2.Template
    prompts: Dict[str, jinja2.Template]
    inputs: Iterable[PromptInput]
    batch_size: int
    verbose: bool
    randomize: bool
    openai_api_org: str
    openai_api_key: str
    openai_temperature: float
    openai_max_tokens: int
    openai_n: int
    openai_n_retries: int
    openai_retry_timeout_s: int
    openai_read_timeout_s: int

    openai_completions_endpoint: ClassVar[str] = OPENAI_DEFAULTS.COMPLETIONS_ENDPOINT
    retry_error_codes: ClassVar[List[int]] = [429, 503]

    def __init__(
        self,
        display: jinja2.Template,
        prompts: Dict[str, jinja2.Template],
        inputs: Iterable[PromptInput],
        *,
        openai_api_org: str,
        openai_api_key: str,
        openai_model: str,
        batch_size: int = 10,
        verbose: bool = False,
        randomize: bool = True,
        openai_temperature: float = 0,
        openai_max_tokens: int = 500,
        openai_n: int = 1,
        openai_n_retries: int = 10,
        openai_retry_timeout_s: int = 1,
        openai_read_timeout_s: int = 30,
        repeat: int = 3,
    ):
        self.display = display
        self.inputs = inputs
        self.prompts = prompts
        self.model = openai_model
        self.batch_size = batch_size
        self.verbose = verbose
        self.openai_api_org = openai_api_org
        self.openai_api_key = openai_api_key
        self.openai_temperature = openai_temperature
        self.openai_max_tokens = openai_max_tokens
        self.openai_n = openai_n
        self.openai_n_retries = openai_n_retries
        self.openai_retry_timeout_s = openai_retry_timeout_s
        self.openai_read_timeout_s = openai_read_timeout_s
        self.randomize = randomize
        self.repeat = repeat

    def __iter__(self) -> Iterable[Dict]:
        for input_batch in batch_sequence(self.inputs, self.batch_size):
            for _ in range(self.repeat):
                response_batch = self._get_response_batch(input_batch)
                for input_, responses in zip(input_batch, response_batch):
                    yield self._make_example(
                        input_.id,
                        self.display.render(**input_.prompt_args),
                        responses,
                        randomize=self.randomize,
                        prompt_args=input_.prompt_args,
                    )

    def on_exit(self, ctrl):
        examples = ctrl.db.get_dataset_examples(ctrl.dataset)
        counts = Counter({k: 0 for k in self.prompts.keys()})
        # Get last example per ID
        for eg in examples:
            selected = eg.get("accept", [])
            if not selected or len(selected) != 1 or eg["answer"] != "accept":
                continue
            counts[selected[0]] += 1
        print("")
        if not counts:
            msg.warn("No answers found", exits=0)
        msg.divider("Evaluation results", icon="emoji")
        # Handle edge case when both are equal:
        nr1, nr2 = counts.most_common(2)
        if nr1[1] == nr2[1]:
            msg.good("It's a draw!")
        else:
            pref, _ = nr1
            msg.good(f"You preferred {pref}")
        rows = [(name, count) for name, count in counts.most_common()]
        msg.table(rows, aligns=("l", "r"))

    def _get_response_batch(self, inputs: List[PromptInput]) -> List[Dict[str, str]]:
        name1, name2 = self._choose_rivals()
        prompts = []
        for input_ in inputs:
            prompts.append(self._get_prompt(name1, input_.prompt_args))
            prompts.append(self._get_prompt(name2, input_.prompt_args))
        if self.verbose:
            for prompt in prompts:
                log("Prompt to OpenAI", details=prompt)
        responses = self._get_responses(prompts)
        assert len(responses) == len(inputs) * 2
        output = []
        # Pair out the responses. There's a fancy
        # zip way to do this but I think that's less
        # readable
        for i in range(0, len(responses), 2):
            output.append({name1: responses[i], name2: responses[i + 1]})
        return output

    def _choose_rivals(self) -> Tuple[str, str]:
        assert len(self.prompts) == 2
        return tuple(sorted(self.prompts.keys()))

    def _get_prompt(self, name: str, args: Dict) -> str:
        return self.prompts[name].render(**args)

    def _get_responses(self, prompts: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Organization": self.openai_api_org,
            "Content-Type": "application/json",
        }
        r = retry(
            lambda: requests.post(
                self.openai_completions_endpoint,
                headers=headers,
                json={
                    "model": self.model,
                    "prompt": prompts,
                    "temperature": self.openai_temperature,
                    "max_tokens": self.openai_max_tokens,
                    "n": self.openai_n,
                },
                timeout=self.openai_read_timeout_s,
            ),
            n=self.openai_n_retries,
            timeout_s=self.openai_retry_timeout_s,
            error_codes=self.retry_error_codes,
        )
        r.raise_for_status()
        responses = r.json()
        return [responses["choices"][i]["text"].strip() for i in range(len(prompts))]

    def _make_example(
        self,
        id_str: str,
        input: str,
        responses: Dict[str, str],
        randomize: bool,
        prompt_args: Dict[str, Any],
    ) -> Dict:
        question = {
            "id": id_str,
            "text": input,
            "options": [],
        }
        response_pairs = list(responses.items())
        if randomize:
            shuffle(response_pairs)
        else:
            response_pairs = list(sorted(response_pairs))
        for name, value in response_pairs:
            question["options"].append({"id": name, "text": value})
        question["meta"] = prompt_args
        return question


def get_api_credentials(model: Optional[str] = None) -> Tuple[str, str]:
    """Obtain API credentials from a .env file"""
    load_dotenv()
    # Fetch and check the key
    api_key = os.getenv(ENV_VARS.OPENAI_KEY)
    if api_key is None:
        m = (
            "Could not find the API key to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, then make it available as "
            "an environment variable 'PRODIGY_OPENAI_KEY', for instance in a .env file."
        )
        msg.fail(m, exits=1)
    # Fetch and check the org
    org = os.getenv(ENV_VARS.OPENAI_ORG)
    if org is None:
        m = (
            "Could not find the organisation to access the openai API. Ensure you have an API key "
            "set up via https://beta.openai.com/account/api-keys, obtain its organization ID 'org-XXX' "
            "via https://beta.openai.com/account/org-settings, then make it available as "
            "an environment variable 'PRODIGY_OPENAI_ORG', for instance in a .env file."
        )
        msg.fail(m, exits=1)

    # Check the access and get a list of available models to verify the model argument (if not None)
    # Even if the model is None, this call is used as a healthcheck to verify access.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": org,
    }
    r = retry(
        lambda: requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
        ),
        n=1,
        timeout_s=1,
    )
    if r.status_code == 422:
        m = (
            "Could not access api.openai.com -- 422 permission denied."
            "Visit https://beta.openai.com/account/api-keys to check your API keys."
        )
        msg.fail(m, exits=1)
    elif r.status_code != 200:
        m = "Error accessing api.openai.com" f"{r.status_code}: {r.text}"
        msg.fail(m, exits=1)

    if model is not None:
        response = r.json()["data"]
        models = [response[i]["id"] for i in range(len(response))]
        if model not in models:
            e = f"The specified model '{model}' is not available. Choices are: {sorted(set(models))}"
            msg.fail(e, exits=1)

    return api_key, org


def read_prompt_examples(
    path: Optional[Path], *, example_class: Type[PromptExample]
) -> List[PromptExample]:
    if path is None:
        return []
    elif path.suffix in (".yml", ".yaml"):
        return read_yaml_examples(path, example_class=example_class)
    elif path.suffix == ".json":
        data = srsly.read_json(path)
        assert isinstance(data, list)
        return [PromptExample(**eg) for eg in data]
    else:
        msg.fail(
            "The --examples-path (-e) parameter expects a .yml, .yaml or .json file.",
            exits=1,
        )


def load_template(path: str) -> jinja2.Template:
    # I know jinja has a lot of complex file loading stuff,
    # but we're not using the inheritance etc that makes
    # that stuff worthwhile.
    p = Path(path)
    if not p.suffix == ".jinja2":
        msg.fail(
            "The --prompt-path (-p) parameter expects a .jinja2 file.",
            exits=1,
        )
    with p.open("r", encoding="utf8") as file_:
        text = file_.read()
    return jinja2.Template(text, undefined=jinja2.DebugUndefined)


def retry(
    call_api: Callable[[], requests.Response],
    n: int,
    timeout_s: int,
    error_codes: List[int] = [429, 503],
) -> requests.Response:
    """Retry a call to the OpenAI API if we get a non-ok status code.
    This function automatically retries a request if it catches a response
    with an error code in `error_codes`. The amount of timeout also increases
    exponentially every time we retry.
    """
    assert n >= 0
    assert timeout_s >= 1
    r = call_api()
    i = -1
    # We don't want to retry on every non-ok status code. Some are about
    # incorrect inputs, etc. and we want to terminate on those.
    while i < n and r.status_code in error_codes:
        time.sleep(timeout_s)
        i += 1
        timeout_s = timeout_s * 2  # Increase timeout everytime you retry
        msg.text(
            f"Retrying call (retries left: {n-i}, timeout: {timeout_s}s). "
            f"Previous call returned: {r.status_code} ({r.reason})"
        )
    return r


def read_yaml_examples(
    path: str, *, example_class: Type[PromptExample]
) -> List[PromptExample]:
    p = Path(path)
    data = srsly.read_yaml(p)
    if not isinstance(data, list):
        msg.fail("Cannot interpret prompt examples from yaml", exits=1)
    assert isinstance(data, list)
    output = [example_class(**eg) for eg in data]
    return output


def batch_sequence(items: Iterable[_ItemT], batch_size: int) -> Iterable[List[_ItemT]]:
    batch = []
    for eg in items:
        batch.append(eg)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_resume_stream(stream: Iterable[Dict], cache: Iterable[Dict]):
    # Get all hashes in the cache
    cache_ids = [eg.get("_input_hash") for eg in cache]
    log(f"Found {len(cache_ids)} hashes in cache")
    # Hash the current stream and return examples not in cache
    hashed_stream = [(set_hashes(eg).get("_input_hash"), eg) for eg in stream]
    resume = [eg for _hash, eg in hashed_stream if _hash not in cache_ids]
    return resume
