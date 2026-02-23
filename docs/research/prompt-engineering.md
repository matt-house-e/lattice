# Prompt Engineering Research

> **Date**: February 2026
> **Context**: Research for Accrue field spec redesign and system prompt overhaul

## Summary

Research across OpenAI cookbook, Clay, Instructor, LangExtract, and structured extraction best practices to inform how Accrue structures field specifications and builds LLM prompts.

---

## OpenAI GPT-4.1 Prompting Guide (Key Findings)

Source: [OpenAI Cookbook - GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)

### Literal instruction following

GPT-4.1 follows instructions more literally than predecessors. GPT-4o would liberally infer intent from vague prompts. GPT-4.1 does exactly what you tell it -- no more, no less. A single sentence firmly clarifying desired behavior is almost always sufficient.

### Recommended system prompt structure

```
# Role and Objective
# Instructions
## Sub-categories for more detailed instructions
# Reasoning Steps
# Output Format
# Examples
## Example 1
# Context
# Final instructions and prompt to think step by step
```

### Prompt format: Markdown + XML hybrid

- **Markdown headers** (`#`, `##`) for top-level sections -- OpenAI's recommended starting point
- **XML tags** for data/content boundaries -- wrapping documents, data blocks, structured content
- **JSON in prompts performed poorly** in OpenAI's long-context testing
- XML "performed well" for structured data delimiters
- Simple `ID: 1 | TITLE: ... | CONTENT: ...` also performed well
- The hybrid approach (markdown sections + XML data) is what OpenAI's own docs use

### Dynamic prompts and caching

- Keep static portions at the beginning (for prompt caching -- OpenAI caches from prompt start)
- Put variable content (user input, dynamic context) at the end
- Sandwich method for long contexts: key instructions both before AND after dynamic content
- Only include sections that are relevant -- GPT-4.1's literal following means irrelevant sections can confuse

### GPT-4.1 family differences

| Model | Best For | Notes |
|-------|----------|-------|
| gpt-4.1 | Demanding tasks, complex reasoning | Slowest, most capable |
| gpt-4.1-mini | General use, structured extraction | "Standout star," matches/beats GPT-4o, 83% cheaper |
| gpt-4.1-nano | Simple classification, autocomplete | Cheapest/fastest, poor on complex retrieval, doesn't support web search |

For Nano: flatter schemas and simpler prompts are critical. Structured outputs may increase hallucinations on smaller models.

---

## Structured Outputs (json_schema vs json_object)

### json_object (legacy)

OpenAI now considers `response_format: {"type": "json_object"}` to be legacy. It only guarantees syntactically valid JSON, not schema adherence. **This is what Accrue currently uses.**

### json_schema + strict: true (recommended)

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "enrichment_result",
        "schema": pydantic_model.model_json_schema(),
        "strict": True
    }
}
```

Uses constrained decoding (context-free grammar token masking) to guarantee output matches schema exactly. Supported in Chat Completions and Responses API.

### Schema design best practices

1. Make schemas as flat as possible -- flatter = more reliable, especially smaller models
2. Include a reasoning/explanation field first -- chain-of-thought within structured output
3. Use enums to reduce hallucinations -- constrained token choices
4. Set `additionalProperties: false` to prevent schema violations
5. Field descriptions complement, don't duplicate, system instructions
6. Descriptive field names function as implicit documentation

### What the model sees

When you pass a Pydantic model as `response_format`, the complete JSON schema is placed into the system message. The model sees all field names, `Field(description=...)` values, type annotations, enums, and nested structures. Field names and descriptions directly improve output quality.

### Limitations

- Max 100 total properties across entire schema
- Max nesting depth of 5 levels
- Combined character count of all property names, definitions, enum values, const values <= 15,000
- `default` is NOT enforced by constrained generation
- No `pattern`, `minLength`, `maxLength`, `minimum`, `maximum` in constrained mode (prompt hints only)

---

## Industry Research: How Enrichment Tools Structure Fields

### Clay.com

Per-column enrichment configuration:
- **Data types**: Text, Number, URL, Checkbox (Boolean), Select (enum), Date, Currency, Email, Image
- **AI column config**: Model selection (GPT/Claude/Gemini), prompt, output fields with name + type, examples, JSON Schema mode for power users
- **Conditional execution**: "Required to Run" toggle -- skip enrichment if input field is empty
- **Waterfall enrichment**: Check multiple providers in priority order, merge results
- **Prompt best practices**: Use second person, provide fallback logic, add token-saving instructions, match model to complexity

### Instructor (jxnl/instructor)

Field-level guidance via Pydantic:
- `Field(description="...")` -- prompt-level guidance (most important)
- `Literal[...]` -- enum constraint
- `ge`, `gt`, `le`, `lt` -- numeric range constraints
- `min_length`, `max_length` -- string/list length constraints
- `pattern` -- regex constraint
- `Optional[T]` -- field can be null/omitted
- `default` / `default_factory` -- value if LLM omits the field
- `examples` via `json_schema_extra` -- injected into JSON schema
- `exclude=True` -- present in validation but excluded from output (chain-of-thought)
- Validators (code-based, LLM-based) trigger retry on failure

### Google LangExtract

- Schema-by-example: provide `ExampleData` objects instead of schema classes
- Source grounding: character-level span mapping back to source text
- Verbatim requirement: extracted text must match source exactly
- Interactive HTML visualizer for audit

### Best Practices (Research Consensus)

| Practice | Finding |
|----------|---------|
| Few-shot examples | Complex extraction benefits significantly; simple classification works with instructions alone |
| Good + bad examples | Both valuable: good shows target, bad shows common mistakes to avoid |
| Default/fallback values | Always specify per-field: what to return when data is insufficient |
| Format constraints | Specify explicitly: date format, number format, max length, max items |
| Enums | Constrained value lists reduce hallucination significantly |
| Source attribution | Best done as separate field, not metadata on other fields |

---

## Web Search Integration (OpenAI Responses API)

### How it works

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{
        "type": "web_search",
        "search_context_size": "medium",  # low | medium | high
    }],
    input="Research company XYZ",
    include=["web_search_call.action.sources"],
)
```

### Citation format

Inline annotations with `url_citation` objects:
```json
{
    "type": "url_citation",
    "url": "https://example.com/source",
    "title": "Page Title",
    "start_index": 42,
    "end_index": 87
}
```

### Web search + structured output = unreliable

Combining `web_search` tool + `json_schema` response format in one call causes:
- Response truncation mid-JSON
- High failure rates (5-10x retries)
- Silent failures (charged for broken responses)

### Recommended pattern: two-step

```python
# Step 1: Web search (plain text + citations)
search_response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "web_search"}],
    input=f"Research {company}..."
)
web_context = search_response.output_text
citations = extract_citations(search_response)

# Step 2: Structured extraction (no web search)
extraction = client.responses.create(
    model="gpt-4.1-mini",
    input=f"Based on this research:\n{web_context}\n\nExtract...",
    text={"format": {"type": "json_schema", ...}}
)
```

### Accrue integration

Maps perfectly to FunctionStep (search) → LLMStep (extract):
```python
Pipeline([
    FunctionStep("research", fn=web_search_fn, fields=["__web_context", "sources"]),
    LLMStep("analyze", fields={...}, depends_on=["research"]),
])
```

### Limitations

- gpt-4.1-nano does NOT support web search
- 128k context cap on web search (even on 1M-window models)
- Web search tool call cost: ~$10/1,000 calls (GA) or ~$25/1,000 (preview)
- Internal sub-searches can inflate costs 2-3x

### Alternative: Tavily API

For users who want web search without OpenAI's Responses API, Tavily is a popular search API that returns structured results. Would be wrapped in a FunctionStep.

---

## Decisions Made From This Research

| Decision | Rationale |
|----------|-----------|
| Default model → `gpt-4.1-mini` | Better quality for enrichment, still cheap. Nano reserved for simple classification. |
| Field spec: 7 keys (prompt, type, format, enum, examples, bad_examples, default) | Covers what Clay/Instructor/research says matters. Merged guidance into prompt. |
| Dynamic system prompt | Only describe keys actually used. Saves tokens, avoids confusing literal gpt-4.1. |
| Markdown headers + XML data boundaries | OpenAI's recommended hybrid pattern. JSON in prompts performs poorly. |
| Three-tier prompt customization | system_prompt (full override), system_prompt_header (inject), default (dynamic). |
| Web search via two-step FunctionStep → LLMStep | Avoids web search + structured output incompatibility. Aligns with FunctionStep escape hatch design. |
| Migrate to Structured Outputs (json_schema + strict) | Legacy json_object only guarantees syntax, not schema. Future work. |
