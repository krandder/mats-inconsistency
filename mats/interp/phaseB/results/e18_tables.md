| arm | model / effort | framing | pairs | shift [95% CI] | pairs + / − | Bayes shift (rounding) | reliability slope | mean abs dev from Bayes | within-prompt sd |
|---|---|---|---|---|---|---|---|---|---|
| S_haiku | Haiku 5.5 / low | stated | 60 | -0.044 [-0.193, +0.104] | 29 / 22 | +0.001 | 0.99 | 0.030 | 0.026 |
| S_sonnet | Sonnet 5.5 / low | stated | 60 | -0.042 [-0.127, +0.042] | 20 / 29 | +0.001 | 1.00 | 0.048 | 0.015 |
| S_opus | Opus 5.5 / low | stated | 60 | +0.022 [-0.031, +0.078] | 20 / 17 | +0.001 | 1.00 | 0.009 | 0.007 |
| S_gpt | GPT-6 / low | stated | 60 | -0.036 [-0.083, +0.002] | 8 / 9 | +0.001 | 1.00 | 0.006 |  |
| R_opus | Opus 5.5 / high | stated | 30 | +0.013 [-0.015, +0.059] | 2 / 2 | +0.001 | 1.00 | 0.004 |  |
| U_sonnet | Sonnet 5.5 / low | unstated | 60 | -1.599 [-3.283, -0.330] | 22 / 29 | +0.001 | 0.92 | 0.407 |  |

| arm | calls | ok | cost (USD, Claude list price) | GPT tokens | seconds | calls that used tools |
|---|---|---|---|---|---|---|
| S_haiku | 240 | 240 | 0.61 | 0 | 5688 | 0 |
| S_sonnet | 250 | 240 | 3.21 | 0 | 2854 | 0 |
| S_opus | 242 | 242 | 9.06 | 0 | 4981 | 0 |
| S_gpt | 122 | 120 | 0.00 | 979426 | 8908 | 2 |
| R_opus | 60 | 60 | 4.32 | 0 | 1939 | 0 |
| U_sonnet | 130 | 120 | 1.01 | 0 | 1027 | 0 |
