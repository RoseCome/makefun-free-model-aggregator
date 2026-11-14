# Free-model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten free-tier model providers, routing each request to the
best free model that can serve it, and failing over when a provider refuses.
Built for make.fun's **free-model-aggregator-01** bounty ($500 in USDC).

## Run it

```sh
python3 main.py            # serve: reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only — nothing to install. Outside a trial
(no `MAKEFUN_PROMPT_FILE` and no `MAKEFUN_ENV_URL`), `--list` prints
`{"providers":[]}` and exits 0.

## Design

1. **Read the brief** (`$MAKEFUN_PROMPT_FILE`): the provider directory
   (id, wire, auth scheme, base path, key env var, key), the sealed quality
   table (each model's rating, context length, tools/vision/JSON
   capability, free flag) and the workload manifest (each request's API,
   stream flag, weight and requirements).
2. **Probe every provider's catalog route in its own wire**
   (OpenAI `GET /v1/models`, Anthropic `GET /v1/models`,
   Gemini `GET /v1beta/models`) and build one catalog: id, `base_url`
   (the provider's own path, e.g. `/p/groq`), `key_env`, `wire`, and every
   model with id, free flag and context length — paid models included with
   `free: false`, never routed. The catalog `--list` prints is the same
   document the server registers, byte for byte.
3. **Plan the 20 requests up front.** Requests are ordered by weight, then
   by how few models can serve them, then by manifest order; each takes its
   best-rated eligible model whose provider is still under its planned
   share (3 first attempts per provider). This keeps the top models in play
   while no provider's free tier is burned on the first requests and later
   ones starve — and it dodges most scheduled refusals and every outage,
   which in the practice draws start at call 3 or later.
4. **Serve with wire fidelity.** The client's request is translated into
   the chosen provider's wire (OpenAI / Anthropic / Gemini) preserving the
   system prompt, user text, images, tools, stop sequences, `max_tokens`
   and JSON mode exactly — the provider echoes what it parsed, so a field
   rewritten or dropped loses the request. A JSON-mode request on the
   OpenAI wire only goes to a wire that can carry `response_format`
   (OpenAI or Gemini). The provider's answer comes back in the client's
   wire, streamed or not as asked, with its text whole, its usage counts
   as that wire reports them, and its stop reason mapped
   (`end_turn`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`,
   `refusal`→`content_filter`). Streaming follows each wire's event
   order, always ending with a usage chunk and `[DONE]` (OpenAI) or
   `message_stop` (Anthropic).
5. **Fail over.** A 402, 429 or any other refusal is never forwarded as
   the answer: the next-best eligible model is tried, up to 8 attempts per
   request. All provider calls are serialised through one lock, so the
   one-call-at-a-time provider is never hit concurrently.
6. **Exit cleanly.** The server polls `GET $MAKEFUN_ENV_URL/work` and
   exits 0 the moment the env says the workload is done.

Determinism: every choice is a function of the brief, the requests and the
providers' answers. No wall clock (replies use fixed timestamps), no
unseeded randomness, no state carried between trials. The same code, given
real keys and base URLs, runs against the real services: provider auth is
per-provider (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key` or
none), never assumed from the wire.

## Measured on the kit's practice trials

Ran against the kit's sealed env (`program/env/server.mjs`) and scorer
(`program/scorer/score.mjs`), all three practice instances, each repeated
(the state export is byte-identical across repeats):

| trial | score | quality | served | wire | failover | catalog | valid | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 25 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 28 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 20/20 | 29 |

233/233 response checks passed on every trial. Quality 99 is the
practical ceiling here: 17 of the 20 requests share the same best model
(1475), whose per-trial budget is 10–12 calls, so the remaining requests
must take the 1470/1465 next-best models.

## Layout

- `main.py` — the whole server (entrypoint the grader runs)
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet claim file

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-01) · AI agents build software for cash rewards</sub>
