# 004: Provider-Agnostic via LLMClient Protocol

**Status:** Accepted
**Date:** 2026-01-15

**Context:** In v0.2, Lattice was hard-coded to OpenAI (`openai.AsyncOpenAI`). Supporting Anthropic or Google required major refactoring. The obvious solution — use litellm as a universal adapter — would add ~30 transitive dependencies, slow imports, and couple Lattice to litellm's versioning and model-string conventions. LangChain was even heavier and opinionated. Users needed multi-provider support without the dependency tax.

**Decision:** Define an `LLMClient` protocol (~30 lines) with a single `create_completion()` method. Ship three adapters: `OpenAIClient` (default, zero config), `AnthropicClient`, `GoogleClient`. OpenAI is the only required provider dependency. Anthropic and Google are optional extras (`pip install accrue[anthropic]`, `lattice[google]`). OpenAI-compatible providers (Ollama, Groq, DeepSeek, vLLM) work via `base_url` shortcut on LLMStep. Custom providers implement the protocol directly (~30 lines of adapter code).

**Alternatives considered:**
- **Stay OpenAI-only** — Limits adoption. Users on Anthropic or Google can't use Lattice without wrapping their own adapter.
- **Use litellm** — ~30 transitive dependencies, slow cold start, magic model strings, couples Lattice to litellm's release cycle. Violates minimal-dependency principle.
- **LangChain integration** — Opinionated framework with incompatible abstractions. Would make Lattice a LangChain plugin rather than a standalone library.
- **Abstract base class instead of protocol** — Forces inheritance, less Pythonic. Protocol is structural typing — any class with the right method signature works, no import needed.
