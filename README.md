# CMCHP — Cross-Model Context Handoff Protocol

A wire format for handing an agent's state from one model to another: hierarchical goals, confidence-scored working memory, open tool calls, behavioural state, and a decision trace holding only the choices that constrained the path. A capable model compresses; a family-aware expander rebuilds a native prompt for the receiver.

The spec is in [`spec/cmchp-v0.1.json`](spec/cmchp-v0.1.json). It is real and I still think it is right.

## The results I published were not results

I shipped this in May 2026 with a benchmark table and a paper reporting 90% mean goal preservation, 85% memory coverage, 93% continuation coherence, and 34x compression. I audited my own repository in September and none of those numbers are reachable by the code that produced them.

Here is what the benchmark actually did. It compressed a conversation into a packet, expanded that packet back into text, and then measured what fraction of the packet's goals appeared in that text. But the text was rendered _from_ the packet. Writing the goals into it is the expander's entire job. So the metric asked whether a renderer renders, and got 100%, and would have got 100% for any input at all:

```python
@pytest.mark.parametrize("goal_text", [
    "migrate the authentication service to short-lived tokens",
    "asdfgh qwerty zxcvbn poiuyt lkjhgf mnbvcx",  # meaningless
    "the quick brown foxes jumped over thirteen lazy sleeping dogs",
])
def test_goal_preservation_scores_100_percent_for_any_goal(goal_text):
    ...
    assert score == 1.0
```

That test passes. `_goal_preservation_score(packet, expanded)` does not take the source conversation as an argument, so no amount of information lost during compression can move the number. `memory_coverage` has the same defect. `continuation_viability`, reported as a success rate, was `len(hint.split()) >= 8` — a word count.

The paper is worse than the README. It reports a replay baseline, an XML-versus-markdown ablation, cross-provider degradation figures, and coherence "as judged by an independent evaluator". None of those exist in the source. The words _XML_, _pruning_ and _replay_ appear in the paper 4, 2 and 6 times; in the code, zero. The README table and the paper table also disagree with each other about the same experiment — two different sets of invented numbers for one run that never happened.

The paper is still in this repository, under a retraction banner, because deleting it would be a second dishonesty. The proof lives in [`tests/test_old_benchmark_was_tautological.py`](tests/test_old_benchmark_was_tautological.py) rather than in prose, because a claim this embarrassing should be executable.

## What replaces it

[`handoff_benchmark.py`](src/cmchp/handoff_benchmark.py) asks a question that can come out wrong.

A model that never saw the source conversation answers probe questions — "how many threads was the process running", "which YC batch was Manifold in" — from three contexts:

| condition   | what the receiving model gets                                                      |
| ----------- | ---------------------------------------------------------------------------------- |
| `full`      | the entire conversation and tool history                                           |
| `packet`    | only the CMCHP packet, expanded                                                    |
| `truncated` | the tail of the raw conversation, cut to the same character count the packet costs |

`full` is the ceiling. **`truncated` is the comparison that matters**: keeping the last N characters is what you do if the protocol does not exist, so a packet that cannot beat it at equal cost has not earned its complexity. The old benchmark had no baseline at all, which is the deeper reason it could not fail.

Ground truth is hand-written in [`benchmarks/probes.json`](benchmarks/probes.json), and a test asserts every probe is answerable from its source conversation — otherwise `full` is not a ceiling and the comparison means nothing. Grading is substring matching, not an LLM judge: a judge would put a second noisy model inside a measurement whose whole point is to be checkable.

## It has not been run

I wrote the harness, unit-tested the grading and the budget arithmetic, and then could not run it: the Anthropic account behind it is out of credits.

So there are no numbers in this README, and there will not be until it runs. That is the honest state, and after the last set of numbers I invented, an empty results section is the only acceptable one.

```bash
uv venv && uv pip install -e ".[dev]"
python -m pytest              # 32 tests, no network needed

export ANTHROPIC_API_KEY=...
python -m cmchp.handoff_benchmark   # writes benchmarks/results.json
```

## What this does not do

It does not serialize an execution graph — no tool-call resumption, no partial state machines. It moves semantic state and stops there.

The compressor is itself an LLM call, so it can lie. Nothing in the protocol verifies that the packet's claims about the conversation are true; the receiving model has no way to tell a faithful compression from a confident fabrication. That is the most serious open problem here and the new benchmark measures its consequences, not its cause.

Confidence scores on memory entries are asserted by the compressing model and are not calibrated against anything.

Three scenarios and roughly six probes each is a small benchmark. Whatever it eventually reports will have intervals wide enough to swallow most of the differences worth caring about — see [error-bars](https://github.com/pvcomms/error-bars), which I wrote after learning that lesson the expensive way on a different project.

## License

MIT.
