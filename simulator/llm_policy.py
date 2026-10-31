"""A policy where a local LLM decides, each session, which stored facts to re-check.

The model sees the user's request and every stored fact with its value and its age
(sessions since last confirmed). It never sees the dependency list, the decision
rule, change rates or the true state. It replies with JSON {"recheck": [fact names]}.

Invalid replies (not JSON, wrong shape, unknown fact names) count as "ask nothing"
and are logged. Calls run at temperature 0 and are cached by prompt hash, so the cache
never changes a decision. The policy is still marked stateful (it holds a call log),
so it is never mixed; other families are tuned to its realized cost instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from simulator.core import Domain, Memory, Task
from simulator.llm import OllamaClient


@dataclass(frozen=True)
class CallRecord:
    """One decision call, for the per-call log."""

    seed: int
    user: int
    session: int
    task: str
    valid: bool
    n_recheck: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cached: bool


PLAIN_INSTRUCTION = (
    "You are a personal assistant. Before answering the user's request, you may ask "
    "the user to re-confirm some stored facts, because facts can change between "
    "sessions without you being told. Each question costs the user time, so only "
    "re-check facts that matter for this request and may be out of date."
)
# Step-by-step (guided) instructions.
GUIDED_INSTRUCTION = "\n".join([
    "Before answering the user's request, decide which stored facts to re-check.",
    "Step 1: list the stored facts this request actually depends on. Ignore facts that "
    "cannot change the answer to this request.",
    "Step 2: for each of those facts, judge from its age (sessions since last confirmed) "
    "how likely it is to have changed. Facts that change often become unreliable quickly; "
    "stable facts stay reliable for a long time.",
    "Step 3: re-check only facts that are both needed for this request and likely to have "
    "changed. Asking costs the user effort, so do not re-check facts that are irrelevant "
    "or recently confirmed.",
])
INSTRUCTIONS = {"plain": PLAIN_INSTRUCTION, "guided": GUIDED_INSTRUCTION}


def build_prompt(domain: Domain, memory: Memory, task: Task, t: int,
                 style: str = "plain") -> str:
    """Prompt listing the request and every stored fact with its value and age; `style`
    selects the plain or guided instructions."""
    lines = []
    for a in domain.attributes:
        age = memory.age(a.name, t)
        value = memory.values[a.name].replace("_", " ")
        lines.append(f"- {a.name}: {value} (last confirmed {age} session"
                     f"{'' if age == 1 else 's'} ago)")
    request = task.description or task.name.replace("_", " ")
    return (
        INSTRUCTIONS[style] + "\n\n"
        f'User request: "{request}"\n\n'
        "Stored facts about the user:\n" + "\n".join(lines) + "\n\n"
        "Which facts should you re-check before answering? Reply with JSON only, in the "
        'form {"recheck": [<fact names exactly as written above>]}; use an empty list '
        "to re-check nothing."
    )


def parse_recheck(text: str, names: set[str]) -> list[str] | None:
    """Fact names to re-check, or None if the reply is invalid."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("recheck"), list):
        return None
    out = obj["recheck"]
    if not all(isinstance(x, str) and x in names for x in out):
        return None
    return list(dict.fromkeys(out))


class LLMPolicy:
    """Per-session LLM re-ask decisions (see module docstring)."""

    role = "sweep"
    stateful = True

    def __init__(self, domain: Domain, client: OllamaClient, prompt_style: str = "plain") -> None:
        if prompt_style not in INSTRUCTIONS:
            raise ValueError(f"unknown prompt style {prompt_style!r}")
        self.domain, self.client, self.prompt_style = domain, client, prompt_style
        tag = "llm" if prompt_style == "plain" else f"llm-{prompt_style}"
        self.name = f"{tag}({client.model})"
        self.family = "llm"
        self.names = {a.name for a in domain.attributes}
        self.calls: list[CallRecord] = []
        self._seed = self._user = -1

    def start_user(self, seed: int, user: int) -> None:
        """Remember whose timeline is running, for the call log."""
        self._seed, self._user = seed, user

    def decide(self, memory: Memory, task: Task, t: int) -> list[str]:
        """Ask the model; invalid replies mean ask nothing."""
        resp = self.client.generate(build_prompt(self.domain, memory, task, t,
                                                 self.prompt_style))
        picked = parse_recheck(resp.text, self.names)
        self.calls.append(CallRecord(
            self._seed, self._user, t, task.name, picked is not None,
            len(picked or []), resp.latency_s, resp.prompt_tokens, resp.completion_tokens,
            resp.cached))
        return picked or []

    def call_log(self) -> list[dict]:
        """Per-call records as dicts."""
        return [asdict(c) for c in self.calls]
