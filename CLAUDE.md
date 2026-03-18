everything needs to use free data and no paid apis

Below is a \*\*PhD-grade “SpecKit / BMAD plan”\*\* for a daily-running system that outputs your own \*\*projected CBB spread + total (O/U)\*\* for every game on the slate, plus \*\*uncertainty + edge vs market + bet sizing\*\*, with an engineering workflow designed to \*\*minimize hallucinations\*\* when vibe-coding with \*\*Claude Code + Opus 4.5\*\* (including the model/tooling constraints that matter).



You \*can’t\* guarantee “unbeatable” (CBB betting markets are often close to efficient in aggregate), but you \*can\* build a research-grade pipeline that’s capable of finding \*\*repeatable micro-edges\*\* and proving (or falsifying) them with \*\*closing-line-value\*\* and robust backtests. (\[ResearchGate]\[1])



---



\## 0) Target deliverables (what “done” looks like)



\*\*Daily (auto-run):\*\*



1\. Pull today’s (or tomorrow’s) schedule.

2\. Produce for each game:



&nbsp;  \* `proj\_spread` (home – away)

&nbsp;  \* `proj\_total`

&nbsp;  \* `win\_prob\_home`

&nbsp;  \* `spread\_ci\_50/80/95` and `total\_ci\_50/80/95`

&nbsp;  \* `edge\_vs\_market\_spread`, `edge\_vs\_market\_total` (if odds available)

&nbsp;  \* `recommended\_bet` with sizing + max-risk guardrails

3\. Log everything (inputs, features, model version, outputs, market snapshot timestamps).



\*\*Weekly (auto-run):\*\*



\* Rolling backtest update, calibration plots, CLV report (by book, time-to-tip, conference, travel, etc.).



\*\*North-star success metrics:\*\*



\* \*\*Beat a sharp close\*\* on average (CLV vs a sharp reference is the primary skill signal). (\[Pinnacle]\[2])

\* Secondary: spread MAE / total MAE vs close, probabilistic scores (NLL / CRPS), and realized ROI with conservative staking.



---



\## 1) Data: minimum viable feeds (and “pro mode” upgrades)



\### 1.1 Game + team + play-by-play (free-ish)



\* \*\*hoopR\*\* (SportsDataverse) can pull men’s CBB schedule, box scores, play-by-play, and ESPN-derived endpoints. (\[hoopr.sportsdataverse.org]\[3])

\* Alternative/backup: \*\*Sportradar NCAAMB API\*\* (paid, reliable real-time, standardized IDs). (\[Getting Started]\[4])



\*\*Why you need PBP eventually:\*\* lineup volatility, foul rates, pace shocks, garbage time, and player-impact models.



\### 1.2 Odds / lines (you need \*historical\* + \*timestamped snapshots\*)



\* \*\*The Odds API\*\* supports NCAAB spreads/totals and offers historical snapshots (with defined availability windows). (\[The Odds API]\[5])

\* (Optional paid upgrades) SportsDataIO historical odds, OddsJam, etc. (\[SportsDataIO]\[6])



\*\*Critical:\*\* store \*\*multiple timestamps\*\* (open, +24h, +6h, +1h, close). CLV analysis depends on it. (\[Pinnacle]\[2])



\### 1.3 “Sharp” reference line



Use the sharpest available close (often \*\*Pinnacle/market-maker style\*\*) as your efficiency benchmark. (\[Pinnacle]\[2])



\### 1.4 Team strength priors (tempo-free efficiency)



Two proven public baselines:



\* \*\*KenPom-style adjusted efficiencies + tempo\*\* (possession-based, opponent-adjusted; possessions estimated with `FGA - OR + TO + 0.475\*FTA`). (\[KenPom]\[7])

\* \*\*T-Rank (Bart Torvik)\*\* also uses adjusted efficiency + tempo to project scores (explicitly described by Torvik). (\[Reddit]\[8])



You can implement your own “KenPom/Torvik-like” core without copying proprietary ratings.



---



\## 2) Modeling architecture (PhD-grade, production-friendly)



\### Big idea



Model each game as:



\* \*\*Possessions\*\* (tempo)

\* \*\*Points per possession\*\* for each team (offense vs opponent defense, plus context)

&nbsp; Then convert to:

\* \*\*Expected scores\*\*, \*\*spread\*\*, \*\*total\*\*, and \*\*uncertainty\*\*.



This mirrors how Torvik describes projecting scores (expected PPP × expected possessions). (\[Reddit]\[8])



---



\## 3) Core math: from team strengths → spread + total



\### 3.1 Notation



For game (g) with home team (H), away team (A):



\* Team latent strengths (time-varying):



&nbsp; \* (O\_t): offensive strength

&nbsp; \* (D\_t): defensive strength

&nbsp; \* (T\_t): tempo preference (possessions/40)

\* Context:



&nbsp; \* home-court effect (h) (global + team-specific optional)

&nbsp; \* travel/rest/altitude/neutral-site flags

&nbsp; \* injury/availability (optional module)



\### 3.2 Expected possessions



A robust starter:

\[

\\mathbb{E}\[P\_g] = \\text{TempoCombine}(T\_H, T\_A) \\times \\text{ContextAdjust}

]

Where `TempoCombine` is typically a \*\*harmonic mean\*\* or \*\*learned convex combination\*\* (learn the mix; don’t hardcode).



Tempo is fundamentally possessions/40, and possessions can be estimated from box score stats as above. (\[KenPom]\[9])



\### 3.3 Expected PPP (matchup)



A strong “tempo-free” starting point (similar spirit to Torvik’s public description):

\[

\\mathbb{E}\[\\text{PPP}\*{H}] = f(O\_H, D\_A, \\text{Context})

]

\[

\\mathbb{E}\[\\text{PPP}\*{A}] = f(O\_A, D\_H, \\text{Context})

]

Torvik has publicly shown a multiplicative baseline using adjusted efficiencies vs average efficiency. (\[Reddit]\[8])



\### 3.4 Expected points



\[

\\mu\_H = \\mathbb{E}\[P\_g] \\cdot \\mathbb{E}\[\\text{PPP}\_H], \\quad

\\mu\_A = \\mathbb{E}\[P\_g] \\cdot \\mathbb{E}\[\\text{PPP}\_A]

]

Then:



\* \*\*Projected spread\*\*: (\\mu\_H - \\mu\_A)

\* \*\*Projected total\*\*: (\\mu\_H + \\mu\_A)



\### 3.5 Distribution / uncertainty (don’t skip this)



You need predictive distributions, not just point estimates.



Two common approaches:



1\. \*\*Bivariate score model\*\* (e.g., Skellam / bivariate Poisson style) — used in sports score modeling literature; margin-of-victory can be modeled via derived distributions. (\[De Gruyter Brill]\[10])

2\. \*\*Gaussian residual model\*\* for spread/total:

&nbsp;  \[

&nbsp;  S\_g \\sim \\mathcal{N}(\\mu\_H-\\mu\_A,\\ \\sigma^2\_{\\text{spread}}(P\_g,\\text{teams}))

&nbsp;  ]

&nbsp;  \[

&nbsp;  U\_g \\sim \\mathcal{N}(\\mu\_H+\\mu\_A,\\ \\sigma^2\_{\\text{total}}(P\_g,\\text{teams}))

&nbsp;  ]

&nbsp;  …and learn (\\sigma) as a function of pace, season phase, and team volatility.



\*\*Calibration upgrade:\*\* conformal prediction intervals (distribution-free) for spread/total bands. (\[arXiv]\[11])



---



\## 4) Feature engineering: what actually moves spreads/totals



\### 4.1 “Four Factors” (tempo-free why)



The four factors are a compact causal-ish decomposition of team quality:



\* eFG%, TO%, ORB%, FTr (plus defensive versions). (\[KenPom]\[12])

&nbsp; These help explain \*why\* your model differs from market and reduce overfitting.



\### 4.2 Schedule/spot/context



\* Home/away/neutral, travel distance, rest days, altitude, “3-in-5” fatigue.

&nbsp; Home-court advantage exists and varies; you can model it hierarchically (global + team random effect). (\[The Sport Journal]\[13])



\### 4.3 Player impact (optional but high edge if done right)



\* Regularized APM/RAPM-style player value (ridge), lineup synergy.

&nbsp; RAPM concepts are widely implemented with ridge regularization. (\[tothemean.com]\[14])

&nbsp; \*\*Practical reality:\*\* NCAA player data is noisy; use priors and shrinkage heavily.



\### 4.4 Market features (use the market without copying it)



Include:



\* opener / current / close (when available),

\* line movement and volatility,

\* book disagreement (dispersion),

&nbsp; as \*features\* or as an anchoring prior (below).



---



\## 5) “Vegas-beating” strategy: market-aware, not market-following



Because markets are often efficient overall, your edge usually comes from:



\* better \*\*uncertainty modeling\*\*

\* better \*\*player availability handling\*\*

\* better \*\*tempo/shot profile matchup effects\*\*

\* faster reaction to information

\* targeting segments where the market is systematically weaker



\### 5.1 Bayesian anchoring to the market (the cleanest way)



Set a prior centered on market, but allow data/model to move you:

\[

\\theta\_g \\sim \\mathcal{N}(L\_g,\\ \\tau^2)

]

\[

y\_g \\mid \\theta\_g \\sim \\mathcal{N}(\\theta\_g,\\ \\sigma^2)

]



\* (L\_g) = market spread or total at a chosen timestamp

\* learn (\\tau) by season phase / conference / info quality

&nbsp; This prevents “hero model” blowups and forces you to prove when you’re right \*against\* the line.



\### 5.2 CLV-first evaluation



Track:



\* \*\*CLV\*\* = (your bet line) – (sharp close) for spread/total

\* segment CLV by bet timing; Pinnacle discusses why early bets can be advantageous for CLV. (\[Pinnacle]\[2])



---



\## 6) Training + validation protocol (no leakage, no BS)



\### 6.1 Rolling origin backtests (mandatory)



\* Train on seasons up to date (t)

\* Predict games on date (t+1)

\* Roll forward daily



\### 6.2 Multiple objective functions



\* Spread MAE (vs actual)

\* Total MAE (vs actual)

\* NLL / CRPS for probabilistic forecasts

\* CLV vs sharp close (primary)

\* Simulated ROI with conservative staking rules



\### 6.3 Season-phase regimes



Early season ≠ conference play ≠ tournament. Model regime shifts explicitly (piecewise priors / time-decay / hierarchical partial pooling).



---



\## 7) System design (repo layout + services)



\### 7.1 Repository structure



```

cbb-lines/

&nbsp; apps/

&nbsp;   api/                 # FastAPI: /slate, /game/{id}, /backtest, /health

&nbsp;   worker/              # daily runner (Prefect / cron / task scheduler)

&nbsp;   dashboard/           # UI (optional)

&nbsp; packages/

&nbsp;   ingest/              # connectors: hoopR/ESPN, odds api, team metadata

&nbsp;   features/            # feature builders + validation

&nbsp;   models/

&nbsp;     core\_bayes/        # tempo + PPP Bayesian model

&nbsp;     residual\_ml/       # XGBoost/CatBoost residuals

&nbsp;     calibration/       # conformal, isotonic

&nbsp;   eval/                # CLV, MAE, ROI sim, calibration plots

&nbsp;   common/              # pydantic schemas, IDs, logging

&nbsp; data/

&nbsp;   raw/                 # immutable snapshots (date-partitioned)

&nbsp;   curated/             # cleaned tables

&nbsp;   model\_store/         # MLflow artifacts / registry

&nbsp; ops/

&nbsp;   docker/

&nbsp;   scripts/

&nbsp;   ci/

&nbsp; docs/

&nbsp;   SPEC.md

&nbsp;   DATA\_CONTRACTS.md

&nbsp;   MODEL\_CARD.md

&nbsp;   RUNBOOK.md

```



\### 7.2 Data contracts (Pydantic everywhere)



Define strict schemas:



\* `Game`, `Team`, `Player`, `LineSnapshot`, `FeatureRow`, `PredictionRow`

&nbsp; Hard-fail if schema mismatches (prevents silent garbage).



\### 7.3 Storage choices



\* DuckDB/Parquet for local MVP

\* Postgres for prod

\* Every odds pull saved as \*\*append-only snapshots\*\*



---



\## 8) “No-hallucination” engineering rules (built for vibe coding)



\### 8.1 Guardrail tooling



\* \*\*Type checking:\*\* mypy/pyright

\* \*\*Runtime validation:\*\* pydantic strict

\* \*\*Unit + integration tests:\*\* pytest

\* \*\*Data validation:\*\* great expectations (optional) or custom checks

\* \*\*Repro:\*\* lockfiles + pinned versions + MLflow model registry



\### 8.2 Failure modes you must explicitly test



\* Team ID mismatches across feeds

\* Neutral site mislabeled as home

\* Duplicated games or rescheduled games

\* Odds missing / stale

\* Partial PBP coverage

\* Garbage-time pollution in player impact stats



---



\## 9) Claude Code + Opus 4.5: how to use it like a pro (and safely)



\### 9.1 Why Opus 4.5 helps here



Anthropic positions Opus 4.5 as strong for “heavy-duty agentic workflows” and coding/refactors. (\[Anthropic]\[15])

And Anthropic’s Claude 4.5 docs note Opus 4.5 uniquely supports an \*\*“effort”\*\* control parameter (useful to force thoroughness on critical modules). (\[Claude]\[16])



\### 9.2 Recommended “BMAD” loop for Claude Code



\*\*BMAD = Build → Measure → Analyze → Deploy\*\*



1\. \*\*Build:\*\* small vertical slice (ingest → features → predict → eval) for 1 day of games

2\. \*\*Measure:\*\* run tests + backtest on a fixed historical week

3\. \*\*Analyze:\*\* error decomposition (pace vs PPP vs HCA vs injuries)

4\. \*\*Deploy:\*\* schedule daily job, freeze model version, log model card



\### 9.3 The Master Prompt you give Claude Code (copy/paste)



```text

You are implementing a production-grade NCAA men’s basketball spread + total modeling system.

Non-negotiable rules:

1\) No unverifiable claims: every data field must come from an implemented connector and schema.

2\) Strict typing + validation: use Pydantic models for all data contracts; fail fast on mismatch.

3\) No silent defaults for missing odds/injuries: propagate “missing” explicitly and gate decisions.

4\) Every new module requires:

&nbsp;  - unit tests

&nbsp;  - integration test that runs a 1-day slate end-to-end

&nbsp;  - a short MODEL\_CARD.md update describing assumptions and limitations

5\) Implement in vertical slices (ingest -> features -> model -> prediction -> eval).

6\) Provide diffs/patches only. After each patch: run tests and show outputs. If tests fail, fix before proceeding.

7\) Any complex math must be documented in docs/SPEC.md with equations and references.

Goal: daily predictions for the slate with uncertainty intervals, edge vs market, and CLV backtesting.

```



\### 9.4 How to split work between Sonnet and Opus



\* Sonnet: connectors, CRUD, UI, boilerplate

\* \*\*Opus (high effort):\*\* core Bayesian model, calibration, backtesting, ID resolution, leakage prevention



---



\## 10) Implementation roadmap (fast path → elite path)



\### Phase 1 (48–72h MVP)



\* Ingest: schedule + final scores (hoopR/ESPN)

\* Ingest: odds snapshots (The Odds API)

\* Build baseline: tempo + PPP model (KenPom/Torvik-like)

\* Output: spread/total + basic intervals

\* Backtest: rolling 30 days



\### Phase 2 (week 1–2)



\* Hierarchical home-court (global + team random effect)

\* Four Factors feature layer (off/def)

\* Market-aware prior (anchoring)

\* CLV reporting dashboard



\### Phase 3 (week 3–6 “edge hunting”)



\* Player impact module (RAPM-lite + shrinkage)

\* Conformal calibration for spread/total intervals

\* Residual ML model stacked on top (e.g., CatBoost)

\* Segment-specific models (conference + non-conference, early season vs late)



---



\## 11) Practical warnings (so you don’t fool yourself)



\* “Beating Vegas” must be defined as \*\*beating a sharp close\*\* and surviving \*\*out-of-sample\*\*. (\[Pinnacle]\[2])

\* NCAA betting market efficiency is hard to beat broadly; your advantage usually comes from being \*better\* on \*\*certain game types\*\*, not all games. (\[ResearchGate]\[1])

\* The system’s real power is the \*\*measurement + iteration loop\*\* (BMAD), not one magic model.



---



\## What I’d do next (if you’re starting today)



1\. Implement \*\*data contracts + ingestion\*\* (schedule, results, odds snapshots).

2\. Implement the \*\*tempo + PPP core\*\* + posterior predictive intervals.

3\. Ship a daily CLI: `predict --date YYYY-MM-DD` that writes `predictions.parquet`.

4\. Add `backtest --start --end` and generate CLV + calibration.



If you want, I can also generate a \*\*fully filled-out `docs/SPEC.md`\*\* (ready to drop into your repo) with:



\* exact schemas,

\* endpoint designs,

\* mathematical derivations,

\* and test plans.



\- \[Reuters](https://www.reuters.com/business/retail-consumer/anthropic-bolsters-ai-model-claudes-coding-agentic-abilities-with-opus-45-2025-11-24/?utm\_source=chatgpt.com)

\- \[Business Insider](https://www.businessinsider.com/claude-opus-4-5-ai-model-anthropic-debut-advanced-features-2025-11?utm\_source=chatgpt.com)

\- \[techradar.com](https://www.techradar.com/ai-platforms-assistants/claude/claude-opus-4-5-is-now-live-and-meaningfully-better-at-everyday-tasks-and-coding-challenges?utm\_source=chatgpt.com)

\- \[tomsguide.com](https://www.tomsguide.com/ai/what-is-claude-everything-you-need-to-know-about-anthropics-ai-powerhouse?utm\_source=chatgpt.com)



\[1]: https://www.researchgate.net/publication/226135869\_Market\_efficiency\_and\_NCAA\_college\_basketball\_gambling?utm\_source=chatgpt.com "Market efficiency and NCAA college basketball gambling"

\[2]: https://www.pinnacle.com/betting-resources/en/educational/efficient-market-hypothesis-in-sports-betting-why-early-bets-beat-the-market?utm\_source=chatgpt.com "Efficient Market Hypothesis in Sports Betting: Why Early Bets Beat ..."

\[3]: https://hoopr.sportsdataverse.org/?utm\_source=chatgpt.com "hoopR • Data and Tools for Men's Basketball • hoopR"

\[4]: https://developer.sportradar.com/basketball/reference/ncaamb-overview?utm\_source=chatgpt.com "NCAA Men's Basketball Overview - Sportradar API"

\[5]: https://the-odds-api.com/sports-odds-data/ncaa-basketball-odds.html?utm\_source=chatgpt.com "NCAA Basketball Odds API"

\[6]: https://sportsdata.io/historical-odds?utm\_source=chatgpt.com "Historical Odds Data - SportsDataIO"

\[7]: https://kenpom.com/blog/ratings-explanation/?utm\_source=chatgpt.com "Ratings Explanation | The kenpom.com blog"

\[8]: https://www.reddit.com/r/CollegeBasketball/comments/dp9fzk/im\_bart\_torvik\_creator\_of\_trank\_barttorvikcom\_ama/?utm\_source=chatgpt.com "I'm Bart Torvik, creator of \\"T-Rank\\" (barttorvik.com), AMA - Reddit"

\[9]: https://kenpom.com/blog/help-with-team-page/?utm\_source=chatgpt.com "Help with team page | The kenpom.com blog"

\[10]: https://www.degruyterbrill.com/document/doi/10.1515/jqas-2014-0055/html?lang=en\&srsltid=AfmBOoq-fl5taoE631Oil5tnP7W9n9thosmYlvMcEFIvbD6X8kakkzdH\&utm\_source=chatgpt.com "A generative model for predicting outcomes in college basketball"

\[11]: https://arxiv.org/pdf/2208.08598?utm\_source=chatgpt.com "\[PDF] Using Conformal Win Probability to Predict the Winners of the ... - arXiv"

\[12]: https://kenpom.com/blog/four-factors/?utm\_source=chatgpt.com "Four Factors | The kenpom.com blog"

\[13]: https://thesportjournal.org/article/the-home-court-advantage-evidence-from-mens-college-basketball/?utm\_source=chatgpt.com "The Home Court Advantage: Evidence from Men's College Basketball"

\[14]: https://www.tothemean.com/2018/10/05/ncaa-rapm.html?utm\_source=chatgpt.com "NCAA Basketball RAPM - tothemean"

\[15]: https://www.anthropic.com/news/claude-opus-4-5?utm\_source=chatgpt.com "Introducing Claude Opus 4.5 - Anthropic"

\[16]: https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-5?utm\_source=chatgpt.com "What's new in Claude 4.5"



