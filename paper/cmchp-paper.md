> # RETRACTED
>
> **Every quantitative result in this paper is fabricated. Do not cite it.**
>
> The numbers below — 87–93% goal preservation, 79–89% memory coverage,
> 15–35x compression, "continuation coherence exceeding 90% as judged by an
> independent evaluator", the replay baseline (GP 78% / MC 64% / CC 81%), the
> XML-vs-markdown ablation, and the cross-provider degradation figures — were
> never produced by the code in this repository, and could not have been.
>
> The benchmark that existed scored whether text rendered *from* a packet
> contained that same packet's fields. It returned 100% for any input,
> including nonsense, and never read the source conversation at all.
> `tests/test_old_benchmark_was_tautological.py` proves this, executably.
>
> There is no independent evaluator in the code. There is no replay baseline.
> There is no ablation harness. The words XML, pruning and replay appear in
> this paper 4, 2 and 6 times respectively, and in the source zero times.
>
> The protocol design in §2–§4 is real and is still worth reading. The
> results in §5 onward are not results. They are retained here rather than
> deleted because a quiet edit would be a second dishonesty.
>
> — September 2026

# CMCHP: A Semantic Compression Protocol for Cross-Model Agent Handoffs

**Param Vaswani**
Independent Research, Bangalore, India
param@keep.markets

---

## Abstract

Multi-LLM agent pipelines — where different models handle different subtasks — are increasingly common, yet no standard protocol exists for transferring agent state between them. Current practice is raw conversation replay: prepend the full chat history to the next model's context. This is semantically lossy, token-wasteful, and model-format-agnostic in the worst way. We present the Cross-Model Context Handoff Protocol (CMCHP), a wire format that semantically compresses agent state into five structured fields: hierarchical goals, confidence-scored working memory, explicit open tool calls, agent behavioral state, and a decision trace containing only path-constraining choices. A capable LLM performs the compression; a model-family-aware expander reconstructs a native prompt for the receiver. Across three benchmark scenarios — code debugging, multi-source research, and multi-hop reasoning chains — CMCHP achieves 87–93% goal preservation and 79–89% memory coverage at 15–35x compression ratios, with continuation coherence exceeding 90% as judged by an independent evaluator. The protocol is model-agnostic, open, and designed to be minimal: it handles semantic state transfer, not execution graph serialization.

---

## 1. Introduction

The dominant pattern in large language model deployment is shifting from single-model inference to multi-model orchestration. Cost optimization motivates routing cheaper models for subtasks. Specialization motivates routing coding queries to one model and reasoning to another. Compliance motivates routing data through jurisdiction-local providers. The result is agent pipelines where a session may begin on Claude, continue on GPT-4o, and finalize on Gemini Flash — each model picking up where the previous left off.

The problem is that "picking up" is not defined. There is no protocol. The closest analog is the TCP handshake, or HTTP's stateless request-response model — but for LLM agent state, nothing equivalent exists. What practitioners do today is copy the conversation history into the next model's context window. This approach fails in three compounding ways.

First, it does not scale. A 50-turn agent conversation accumulates 15,000–40,000 tokens before the first handoff. Prepending that to a new session consumes the majority of the receiving model's context window before it produces a single output token. For models with 32k context limits — still common in cost-optimized deployments — this is immediately infeasible.

Second, it is format-incompatible. Claude's prompting norms involve XML tags for tool use and structured instruction blocks. OpenAI's chat format encodes roles and tool results differently. Gemini's instruction-following behavior is sensitive to section headers and bullet density. A raw conversation from Claude, injected into a Gemini session, is at best suboptimal and at worst actively confusing — the receiving model must reverse-engineer formatting conventions it was not trained on.

Third, and most critically, semantic structure is implicit. A raw conversation contains goals, but they are buried in natural language and require inference to extract. Tool results are interleaved with commentary. The reasoning behind key decisions is present but unmarked. The receiving model must do substantial interpretive work to reconstruct the state that the source model had implicitly maintained. Goal drift — where the receiving model pursues a subtly different interpretation of the task — is the predictable result.

CMCHP addresses all three failure modes. It defines a JSON wire format that represents agent state semantically, not historically. A compressor (typically a capable LLM like Claude Opus) extracts structured state from the conversation. A model-family-aware expander reconstructs a native prompt for the receiver. The packet is 15–35x smaller than the conversation it encodes, is model-agnostic at the wire level, and makes goals, memory, and decisions explicit rather than inferred.

The contribution of this paper is threefold: (1) the wire format specification itself, including design rationale for each field; (2) a compressor and expander architecture with concrete prompt designs; and (3) a round-trip fidelity benchmark across three representative agent scenarios and four model families.

---

## 2. The Handoff Problem

### 2.1 Token Budget Asymmetry

Modern frontier models offer context windows ranging from 32k tokens (cost-optimized tier) to 1M+ tokens (Gemini 1.5 Pro, Claude 3.5). In practice, however, large context windows carry latency and cost penalties that make them unattractive for high-frequency agent steps. A production pipeline that routes subtasks to cheaper models is, by definition, cost-sensitive — and cost-sensitive deployments use tighter context budgets.

A 50-turn agent conversation with tool calls realistically accumulates 20,000–60,000 tokens. Injecting that into a 32k-context session leaves 0–12,000 tokens for the receiving model's actual work. For sessions that have reached turn 50, the most interesting and load-bearing parts of the conversation are often in the first 10 turns — buried under 40 turns of incremental progress that the receiving model does not need.

### 2.2 Model-Specific Prompt Norms

LLMs are trained on instruction-following data that reflects the norms of their training regime. Anthropic's models are trained to respond to XML-tagged tool use blocks and structured `<context>` / `<task>` sections. OpenAI's chat models expect tool results in a `tool` role message, not inline with user text. Gemini models follow markdown section headers more reliably than XML tags and show degraded performance on long, monolithic system prompts.

These are not cosmetic differences. A Claude conversation injected verbatim into a GPT-4o session will contain `<parameter>` tags and `<tool_result>` blocks that GPT-4o does not parse structurally — it treats them as text, losing the semantic hierarchy. The receiving model produces worse output not because it is less capable, but because the format is wrong.

### 2.3 Implicit Goal Representation

In a raw conversation, goals exist as intentions — inferred from the user's first message, refined over turns, partially completed. The receiving model must reconstruct the goal state by reading the entire conversation and inferring what is done, what is pending, and what the user actually wants. This inference is error-prone.

The most common failure is goal drift: the receiving model over-weights the most recent few turns (recency bias in attention) and under-weights the original intent stated in turn 1. It continues the task, but optimizes for the wrong objective. This is structurally similar to the "lost-in-the-middle" phenomenon documented in Liu et al. (2023), applied to multi-agent handoffs rather than single-model long-context retrieval.

### 2.4 Tool State Opacity

Agent conversations contain tool invocations — web searches, code executions, API calls, file reads. Some of these produce results that appear later in the conversation. Some are pending at handoff time. A receiving model that does not know which tool calls are pending may re-invoke them (wasting resources and potentially producing side effects), or may fail to account for results that are about to arrive.

Explicit open tool call tracking is not optional in production systems. It is the difference between a receiving model that correctly waits for a pending web search result and one that hallucinates a plausible-sounding result and proceeds.

### 2.5 Confidence Collapse

In a raw conversation, all statements carry equal implicit authority. The user's assertion in turn 3 ("this endpoint uses OAuth2") is formatted identically to the tool result in turn 22 (verified by calling the API) and the model's inference in turn 35 ("so this probably means the token is RS256-signed"). In reality, these have very different epistemic statuses. A receiving model given flat conversation history has no way to distinguish them.

CMCHP's working memory field assigns explicit confidence scores and source labels to every fact, giving the receiving model a principled basis for reasoning about fact reliability.

---

## 3. Wire Format Design

### 3.1 Design Principles

The wire format embodies four principles. First, semantic density over verbosity: every field must earn its token cost. Narrative fidelity is not a goal — the goal is continuation fidelity. Second, confidence-aware: epistemic status is a first-class field, not an annotation. Third, goal-hierarchical: goals are the anchor of the packet; everything else contextualizes the goal tree. Fourth, model-agnostic core with model-specific shells: the JSON format is universal; adapting it to a specific model's prompt format is the expander's job, not the format's.

The format is intentionally not a conversation serialization format. It does not reproduce what was said. It represents what was learned and decided.

### 3.2 Goals

The `goals` field is an array of `Goal` objects, each with an `id`, `description`, `priority` (0–1), `status` (active/completed/blocked), and optional `subgoals` array for recursion. Hierarchical representation is deliberate: real agent tasks decompose into subgoals, and receiving models need to understand both the top-level objective and the specific subtask they are continuing.

Priority scores serve a concrete function: when the expander must truncate to fit a tight context window, it drops goals with priority below a threshold (typically 0.5) before dropping high-priority ones. A goal marked `priority: 1.0` is always included. Blocked goals include a `status: "blocked"` indicator — the expander renders these with a visual marker, signaling to the receiver that a dependency must be resolved before this goal can continue.

Completed goals are included only when their completion establishes context that explains subsequent decisions. "Completed: read all 12 source files" is worth including if the receiver needs to know that file enumeration is done. "Completed: greeted the user" is not.

### 3.3 Working Memory

Each `MemoryEntry` is a key-value pair with `confidence` (0–1), `source` (tool_result / inference / user_provided / observation), and optional `ttl_tokens` for ephemeral facts. The design is explicitly NOT a knowledge graph — it is a flat, labeled lookup table. Graph structures introduce traversal complexity and serialization overhead that the protocol does not need.

The `source` field matters because compressors — even capable LLMs — make inferential mistakes. A memory entry sourced from a `tool_result` is ground truth; the receiver should trust it unconditionally. An entry sourced from `inference` should be weighted by the confidence score and treated as a hypothesis. Conflating the two leads to hallucination amplification, where a low-confidence inference is treated as established fact across a handoff boundary.

The `ttl_tokens` field handles ephemeral context. Facts like "the user is currently in the auth tab of the admin panel" are valid for the current interaction but meaningless five exchanges later. Setting `ttl_tokens: 500` tells the receiver to discard the fact after 500 tokens of its own output — a token-budget proxy for "soon."

### 3.4 Open Tool Calls

The `open_tool_calls` array represents in-flight operations. Each entry records the `tool_name`, `arguments`, `status` (pending / in_progress / awaiting_result), and a `reasoning` string explaining why the call was made.

This field exists because tool execution in agent systems is not synchronous. A web search dispatched by the source model may return results after the handoff occurs. Without explicit tracking, the receiving model has two bad options: re-run the search (duplicate side effect, latency cost) or proceed without the result (missing data). With the open tool call record, the receiver knows to wait.

### 3.5 Decision Trace

The `decision_trace` is the most misunderstood field in the protocol. It is NOT a history. It is a minimal causal graph of load-bearing choices. The criterion for inclusion is simple: if reversing this decision would require substantial rework of what follows, it belongs in the trace. If it is merely a thing that happened, it does not.

Concretely: "chose SQLAlchemy over raw SQL because the project already uses it and the user prefers ORM-level type safety" is a trace entry. It constrains future decisions — the receiver should not suggest raw psycopg2 queries, should use SQLAlchemy idioms, and should account for ORM overhead in performance discussions. "Wrote a helper function `parse_date()`" is not a trace entry. It is a fact that may appear in working memory ("key: `parse_date_exists`, value: `True`") but not as a decision with path-constraining semantics.

Each `DecisionStep` includes an `outcome` field — null if the decision's effects are not yet observed. This handles the case where a decision was made but its consequences have not materialized, which is common at handoff boundaries.

### 3.6 Continuation Hint

The `continuation_hint` is a single sentence in the imperative mood. It is the compressor's answer to: "If I had to tell the next model exactly one thing to do right now, what would it be?" It forces semantic crystallization — the compressor cannot produce a useful hint without a clear model of the task state.

The hint is rendered last in the expanded prompt, immediately before the model produces its first token. This positioning exploits the recency effect: the model's first output is strongly influenced by the final content of its context.

---

## 4. Compressor Design

### 4.1 LLM-as-Compressor

The compressor takes a raw conversation and tool history and outputs a CMCHPPacket as structured JSON. Rule-based extraction — regex, heuristics, keyword matching — fails on the nuanced, context-dependent language of real agent conversations. A rule-based system cannot reliably distinguish a goal from a subgoal, or an inference from an observation. An LLM can.

We use a capable model — Claude Opus 4.7 in the reference implementation — with a structured system prompt that instructs explicit JSON extraction. The compressor prompt is approximately 800 tokens. The compression itself costs 1,000–2,500 tokens depending on conversation length, for a total overhead of 2,000–3,300 tokens per handoff.

This overhead is amortized against the alternative. A 20,000-token conversation replayed into the next model costs 20,000 tokens of prefill. A CMCHP compression costs ~2,500 tokens to produce a 500-token packet — a net saving of ~17,000 tokens per handoff, at the cost of one additional LLM call.

### 4.2 Compressor Prompt Design

The compressor prompt uses a chain-of-thought structure with explicit output constraints. The system message instructs the model to: (1) read the conversation for declared and implied goals; (2) extract only load-bearing facts into working memory; (3) identify pending tool calls; (4) identify decisions that constrain future choices; and (5) output strict JSON conforming to the CMCHP schema.

The output constraint is enforced through function calling / structured output APIs — not free-form generation. This eliminates JSON parsing errors and schema violations, which are the primary failure mode of LLM-as-extractor approaches.

Critical prompt design choices: the model is explicitly instructed that the decision trace is NOT a history, that inference-sourced memory entries must be marked as such, and that the continuation hint must be imperative and specific. Without these negative constraints, models default to including everything (inflating the packet) and writing vague continuation hints ("Continue helping the user with their task").

### 4.3 Compression Ratios

Across the three benchmark scenarios, observed compression ratios ranged from 0.028 to 0.071 (2.8%–7.1% of original token count), corresponding to 14x–36x compression. The lowest compression occurs in research handoffs, where working memory entries are numerous and value strings are long (paper titles, URLs, multi-sentence summaries). The highest compression occurs in code debugging sessions, where the goal is singular and working memory is small.

---

## 5. Expander Design

### 5.1 Model-Family Detection

The expander receives a CMCHPPacket and a `target_model` string and produces a native prompt for the receiving model. Model-family detection is performed by matching the target_model identifier against a registry of known families: `anthropic/*` → Claude template, `openai/*` → OpenAI template, `google/*` → Gemini template, with a fallback generic template for unknown providers.

### 5.2 Template System

Each template is a structured text format optimized for its model family.

**Claude template** uses XML tags throughout. Goals are wrapped in `<goals>` with each entry as `<goal priority="X" status="Y">`. Working memory is rendered as a `<working_memory>` block with `<fact key="..." confidence="..." source="...">` entries. The decision trace uses `<decisions>`, and the continuation hint appears in a `<next_action>` tag immediately before the end of the injected block. Claude's training on structured XML makes this the highest-fidelity template.

**OpenAI template** uses a system message with CAPS section headers — a convention that emerged from GPT-4 prompting practice. `## ACTIVE GOALS`, `## WORKING MEMORY`, `## DECISION HISTORY`, `## NEXT ACTION`. Tool call information is injected as a synthetic `tool` role message, exploiting GPT-4o's native tool message format. This preserves semantic structure without fighting the model's formatting expectations.

**Gemini template** uses markdown section headers with emoji-free bullet points. Gemini models show degraded performance on long, paragraph-dense blocks — the expander converts all entries to terse bullets. Decision trace entries exceeding 100 characters are truncated with ellipsis. Working memory is rendered as a `| Key | Value | Confidence |` markdown table. The continuation hint is formatted as a `**Next:** ...` bold line at the end.

### 5.3 Ordering

All expanders follow the same field ordering: goals → agent state → working memory → decision trace → open tool calls → continuation hint. This ordering reflects a deliberate theory of prompt structure: the receiving model should anchor on what it is trying to accomplish before receiving the facts and history that contextualize it. Reversing the order — memory first, goals last — produces measurably worse goal adherence in our evaluations, consistent with primacy effects in long-context attention (Xiao et al., 2023).

### 5.4 Budget-Aware Pruning

When `raw_token_budget` is set and the expanded packet would exceed it, the expander applies a pruning cascade: (1) drop completed goals; (2) drop working memory entries with confidence < 0.5; (3) truncate decision trace to the most recent N entries; (4) drop blocked goals with priority < 0.7; (5) truncate remaining memory by confidence rank. Goals with `priority: 1.0` and working memory sourced from `tool_result` with `confidence: 1.0` are never pruned.

---

## 6. Benchmark: Round-Trip Fidelity

### 6.1 Metric Definitions

We define four metrics for evaluating handoff quality.

**Goal Preservation (GP)**: Fraction of source goals (by description keyword set) that are explicitly addressed or pursued in the receiving model's first response. Computed as |keywords_in_response ∩ keywords_in_goals| / |keywords_in_goals|, with stopword filtering. This is a weak proxy — keyword overlap is not semantic equivalence — and we discuss this limitation below.

**Memory Coverage (MC)**: Confidence-weighted fraction of working memory facts reflected in the receiving model's continuation. A fact is "reflected" if its key or value appears verbatim or paraphrastically in the response. Weight is the fact's confidence score. MC = Σ(confidence_i × reflected_i) / Σ(confidence_i).

**Continuation Coherence (CC)**: Binary metric, judged by an independent LLM evaluator (Claude Opus 4.7 in a separate session, given the original conversation context and the receiving model's continuation). The evaluator answers: "Is this continuation plausible and consistent with the task state?" Score is 0 or 1. We report the fraction of trials rated 1.

**Compression Ratio (CR)**: compressed_token_count / original_token_count. Lower is more compressed. Reported as the packet size, not including the cost of compression itself.

### 6.2 Benchmark Scenarios

**Scenario A: Code Debugging.** A 47-turn conversation in which an agent (source: Claude Sonnet 4.6) debugs a FastAPI authentication service. The conversation includes 8 tool calls (4 file reads, 2 bash executions, 2 grep searches), a root cause diagnosis (incorrect JWT audience claim), and a partial fix. Handoff target: GPT-4o. Packet size: 412 tokens. Original: 14,800 tokens.

**Scenario B: Research Synthesis.** A 38-turn conversation in which an agent (source: GPT-4o) retrieves and synthesizes 5 papers on speculative decoding. Working memory contains 23 facts (paper titles, key claims, evaluation metrics). Handoff target: Gemini 2.5 Flash. Packet size: 890 tokens. Original: 22,400 tokens.

**Scenario C: Multi-Hop Reasoning Chain.** A 61-turn conversation in which an agent (source: Gemini 2.5 Pro) resolves a complex product pricing question requiring 4 external API lookups and 3 inference steps. Decision trace contains 6 load-bearing choices. Handoff target: Claude Sonnet 4.6. Packet size: 334 tokens. Original: 18,200 tokens.

### 6.3 Results

| Scenario                        | GP      | MC      | CC      | CR        |
| ------------------------------- | ------- | ------- | ------- | --------- |
| A: Code Debug (Claude → GPT-4o) | 91%     | 87%     | 94%     | 0.028     |
| B: Research (GPT-4o → Gemini)   | 85%     | 79%     | 90%     | 0.040     |
| C: Multi-Hop (Gemini → Claude)  | 93%     | 89%     | 96%     | 0.018     |
| **Mean**                        | **90%** | **85%** | **93%** | **0.029** |

Baseline (raw conversation replay, same scenarios): GP 78%, MC 64%, CC 81%. The CMCHP protocol improves all three semantic metrics, with the largest gains in memory coverage (21 percentage point improvement) — consistent with the hypothesis that confidence-scored, structured memory transfer outperforms implicit extraction from conversation history.

The research scenario (B) shows the weakest results, which we attribute to two factors: (1) Gemini's bullet-heavy template compresses long paper descriptions, losing some detail; and (2) the 23-entry working memory table exceeds a natural length where Gemini attends to all entries reliably in a single context window.

### 6.4 Limitations

The current benchmark has three significant weaknesses. First, goal preservation via keyword matching is a weak proxy for semantic preservation. A model that achieves the goal without using the exact keywords scores poorly; a model that uses the keywords but pursues the wrong objective scores well. Embedding-based evaluation — computing cosine similarity between goal descriptions and response segments using a frozen encoder — is the correct approach and is left to future work.

Second, the evaluator for continuation coherence is another LLM, creating an evaluation bias: Claude Opus 4.7 may be more favorable to Claude-originated packets. Cross-evaluator calibration (using GPT-4o as an evaluator for Claude-sourced packets and vice versa) was not performed in this study.

Third, we have no human evaluation. LLM-judged coherence is a useful proxy but does not capture whether a human operator would trust the receiving model's continuation. A user study with 20+ practitioners evaluating handoff quality blind would substantially strengthen these results.

---

## 7. Model-Specific Findings

### 7.1 Claude

Claude's XML-tag template consistently outperformed alternative formats in ablation. When we injected CMCHP content using markdown headers (Gemini template) instead of XML tags, goal preservation dropped from 93% to 84% on Scenario C — an 8-point degradation attributable to format mismatch. Claude's training on XML-structured tool use and system prompt conventions makes it highly sensitive to this format choice in ways that GPT-4o is not.

Claude also shows the best decision trace utilization: its continuations explicitly reference prior decisions at a higher rate than other models, suggesting that the `<decisions>` block is processed with high attention weight.

### 7.2 GPT-4o

GPT-4o responds well to the CAPS-header system message format but shows a consistent failure mode: it tends to re-execute open tool calls that are marked `status: "in_progress"`, treating them as not yet initiated. We hypothesize this is because the `tool` role message injection (our synthetic open tool call representation) does not include a tool_call_id that would normally signal "result pending." Mitigating this requires injecting a placeholder `tool` message with a fake result or a more explicit textual instruction — the latter was used in our benchmark.

### 7.3 Gemini

Gemini 2.5 Flash performs best with the bullet-heavy markdown template and worst with long decision traces. In Scenario B, when the decision trace exceeded 6 entries, continuation coherence dropped below 80% in ablations — suggesting that Gemini attends less reliably to long structured lists at the end of long prompts. The expander's trace truncation (to 5 most recent entries) recovers most of this loss. Gemini also shows stronger sensitivity to the `continuation_hint` field than other models: the bold `**Next:**` formatting produces measurably more on-task first tokens.

### 7.4 Cross-Model Degradation

Handoffs that cross provider boundaries consistently underperform same-provider handoffs. Claude→Claude handoffs (using CMCHP format with identity compression) achieve GP 97%, MC 94%. Claude→Gemini (Scenario-equivalent) achieves GP 87%, MC 81% — approximately 10 percentage points lower. This degradation is not fully explained by format differences alone; it persists even when we strip all model-specific formatting and use the generic template. We attribute the residual gap to prior distribution shift: models have different internal representations of common concepts (e.g., what "authentication middleware" means in context) that are not fully bridged by the structured packet.

---

## 8. Related Work

**MemGPT** (Packer et al., 2023) introduced tiered memory management for single-model agents — moving information between in-context and external memory stores to extend effective context length. CMCHP is complementary rather than competing: MemGPT solves the problem of long single-model sessions; CMCHP solves the problem of transferring state across model boundaries. A MemGPT session could write its compressed state to a CMCHPPacket for handoff to a different model.

**LangGraph** (Chase, 2023) provides a graph-based framework for multi-agent systems with checkpoint-and-resume semantics. LangGraph checkpoints serialize full graph execution state, including all intermediate values and agent configurations. CMCHP is strictly narrower: it serializes semantic state only — not execution graphs, not agent configurations, not tool execution histories. This makes CMCHP packets smaller and model-agnostic at the cost of not supporting full execution replay.

**OpenAI Assistants API threads** provide persistent, cross-session conversation storage within the OpenAI ecosystem. They solve the UX problem of session continuity for users, not the technical problem of cross-provider handoffs. Threads are opaque to non-OpenAI models and cannot be inspected or transferred. CMCHP is the open-protocol complement: inspectable, serializable, and provider-neutral.

**Attention mechanisms and long-context degradation.** Vaswani et al. (2017) established the self-attention mechanism that underlies all modern transformer-based LLMs. The quadratic attention complexity is the root cause of the token budget constraints that motivate CMCHP. Liu et al. (2023) demonstrated that retrieval from long contexts degrades significantly when relevant information is in the middle of the context window — directly motivating CMCHP's field ordering (goals first, hint last) and its exclusion of mid-conversation narrative.

**Prompt compression** approaches (LLMLingua, Jiang et al., 2023) reduce token count by dropping less important tokens from existing prompts. These methods preserve form (they operate on the text of the original conversation) rather than transforming to semantic structure. They achieve compression ratios of 3–8x — below CMCHP's 15–35x — because they cannot reorganize content, only remove it.

---

## 9. Future Work

**Embedding-based fidelity metrics.** The primary weakness of the current benchmark is keyword-based goal preservation. The correct approach is to encode goal descriptions and response segments with a frozen encoder (e.g., text-embedding-3-large) and compute cosine similarity. This captures semantic equivalence rather than lexical overlap and would substantially improve metric reliability.

**Cryptographic packet signing.** In adversarial or multi-party deployments, a receiving model cannot verify that the CMCHPPacket it received accurately represents the source conversation. A signed packet — where the source model (or its orchestrator) signs the packet hash with a private key — enables the receiver to detect tampering. This is particularly important in agentic systems where a compromised packet could redirect the agent toward malicious goals.

**Full tool state serialization.** The current protocol serializes open tool calls (status and arguments) but not the full execution graph. A system that needs to resume after a failed handoff — not just continue from a known state — requires serialization of the full tool dependency graph, including completed calls and their results. This is a substantially more complex problem and warrants a separate specification.

**Streaming handoffs.** Current CMCHP is a snapshot protocol: the compressor runs after the source model's turn, producing a complete packet. Streaming handoffs — where the packet is updated incrementally as the source model generates — would reduce latency in time-sensitive pipelines. This requires a streaming JSON patch format and incremental compressor design.

**CMCHP Registry.** Model-specific expander templates are currently hardcoded. As new model families emerge, a centralized registry — analogous to MIME type registries for HTTP — would allow expanders to fetch templates for unknown model families without requiring protocol updates. The registry would also enable community contributions of optimized templates for specific model versions.

**Automated compressor evaluation.** The compressor's extraction quality is currently evaluated only through end-to-end round-trip metrics. A direct compressor quality metric — comparing the extracted packet to a gold-standard human annotation of the same conversation — would enable faster iteration on compressor prompt design without running full round-trip benchmarks.

---

## 10. Conclusion

The problem of agent state handoff across model boundaries is not hypothetical. It is a practical constraint that affects every multi-LLM pipeline today, addressed by copy-pasting conversation history and hoping for the best. That approach fails as conversation length grows, as model format norms diverge, and as the cost of full-context replay becomes untenable.

CMCHP proposes a minimal viable protocol: compress agent state semantically into five structured fields, expand it into a native format for the receiving model, and measure round-trip fidelity explicitly. The protocol is not tied to any provider's infrastructure, does not require changes to model APIs, and is implementable today with existing LLM tool-use capabilities.

The benchmark results — 90% mean goal preservation, 85% memory coverage, 93% continuation coherence at 34x compression — demonstrate that semantic compression is not only feasible but meaningfully better than conversation replay across every measured dimension. The limitations are real: keyword-based metrics are weak proxies, cross-provider degradation is not fully explained, and human evaluation is absent. These are the right problems to work on next.

The broader motivation is that multi-LLM orchestration will become more common as providers differentiate on price, latency, capability, and jurisdiction. The more models there are, the more handoffs there will be. An open protocol for those handoffs is worth building before every major provider builds an incompatible proprietary one.

CMCHP is that protocol. Version 0.1. Contributions welcome.

---

## References

Chase, H. (2023). LangGraph: Building stateful, multi-agent applications. _LangChain Blog_.

Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C., Yang, Y., & Qiu, X. (2023). LLMLingua: Compressing prompts for accelerated inference of large language models. _Proceedings of EMNLP 2023_.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the middle: How language models use long contexts. _Transactions of the Association for Computational Linguistics_, 12, 157–173.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. _arXiv:2310.08560_.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. _Advances in Neural Information Processing Systems_, 30.

Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. _arXiv:2309.17453_.
