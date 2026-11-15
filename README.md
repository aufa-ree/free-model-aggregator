# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/aufa-ree/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/aufa-ree/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ranked free AI model aggregator: one Node 22 server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten free-tier providers, routing each request to the best free model that meets it, with failover when a provider refuses. Scores 99/100 across all three practice trials (quality 99, served 100, wire 100, catalog 100, failover 100).

## Run it

```sh
node main.ts --list    # print the catalog of free models, endpoints and keys
node main.ts           # serve the OpenAI + Anthropic APIs (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE from a trial)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=e6ntc21f&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
