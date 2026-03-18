# PhD-Level College Basketball Prediction Metrics Guide

## Executive Summary

This document catalogs ALL advanced metrics, factors, and methodologies required for a production-grade NCAA basketball spread prediction system. Based on comprehensive research across academic papers, industry-standard analytics platforms, and PhD-level statistical methods.

---

## 1. TEAM EFFICIENCY METRICS (Core Foundation)

### 1.1 Adjusted Efficiency (KenPom/Torvik Style)

| Metric | Formula | Description |
|--------|---------|-------------|
| **AdjO** | `RawO × (NatAvg / OppAdjD)` | Adjusted Offensive Efficiency (pts/100 poss) |
| **AdjD** | `RawD × (NatAvg / OppAdjO)` | Adjusted Defensive Efficiency |
| **AdjEM** | `AdjO - AdjD` | Efficiency Margin (the KEY ranking metric) |
| **AdjT** | `RawTempo × (NatAvgTempo / OppAdjT)` | Adjusted Tempo (possessions/40 min) |

**Source:** [KenPom Ratings Methodology](https://kenpom.com/blog/ratings-methodology-update/)

### 1.2 Possession Calculation

```
Possessions = FGA - OR + TO + 0.475 × FTA
```

- **0.475 factor** accounts for and-ones and technical fouls
- Invented by Dean Oliver

**Source:** [Basketball-Reference Four Factors](https://www.basketball-reference.com/about/factors.html)

### 1.3 Pythagorean Win Probability

```
Win% = AdjO^k / (AdjO^k + AdjD^k)

where k = 10.25 (KenPom's optimal exponent for college basketball)
```

**Source:** [KenPom Ratings Explanation](https://kenpom.com/blog/ratings-explanation/)

---

## 2. DEAN OLIVER'S FOUR FACTORS (Must Implement!)

### Updated Weights (2023 Research)

| Factor | Formula | Original Weight | Updated Weight |
|--------|---------|-----------------|----------------|
| **eFG%** | `(FG + 0.5×3P) / FGA` | 40% | **46%** |
| **TOV%** | `TO / (FGA + 0.44×FTA + TO)` | 25% | **35%** |
| **ORB%** | `OR / (OR + Opp_DR)` | 20% | **12%** |
| **FTR** | `FTA / FGA` | 15% | **7%** |

**Key Finding:** eFG% and TOV% are MORE important than Oliver originally thought!

**Sources:**
- [Dean Oliver's Four Factors Revisited](https://arxiv.org/abs/2305.13032)
- [Squared Statistics Introduction](https://squared2020.com/2017/09/05/introduction-to-olivers-four-factors/)
- [Basketball-Reference Four Factors](https://www.basketball-reference.com/about/factors.html)

### Why Four Factors Matter

These factors explain **98% of variance** in offensive efficiency. Our current model only uses raw efficiency—missing the compositional insight!

---

## 3. SHOT QUALITY METRICS

### 3.1 Expected Field Goal Percentage (xFG%)

Shot location-based expected make rate. Key insight: **Midrange shots are inefficient**.

| Shot Zone | eFG% (NCAA Avg) | Efficiency |
|-----------|-----------------|------------|
| At rim (<6ft) | 60-65% | **Best** |
| Short midrange | 38-42% | Worst |
| Long midrange | 35-40% | Bad |
| 3-point | 33-35% | **Good** (1.0-1.05 pts/shot) |

**Key Insight:** Teams that shoot lots of midrange jumpers are LESS efficient.

### 3.2 Three-Point Shooting Variance

**Critical for prediction:** 3PT% has HIGH variance and regresses to the mean!

- A team's 3PT% from last season explains only **~25%** of this season's 3PT%
- Free throw shooting is much more stable (98% skill-based)
- **Implication:** Weight recent 3PT% less heavily than eFG% from 2-point shots

**Source:** [The Power Rank - Predictability vs Skill](https://thepowerrank.com/2020/07/28/predictability-vs-skill-in-sports-analytics-3-point-shooting/)

### 3.3 Shot Quality Platforms

- **ShotQuality** - Uses computer vision for expected points
- **EvanMiya** - Scoring Volume metric
- **CBBAnalytics** - Shot charts by location

---

## 4. PLAYER-LEVEL METRICS

### 4.1 Plus-Minus Family

| Metric | Method | Accuracy |
|--------|--------|----------|
| **Raw +/-** | Simple point differential | Low |
| **APM** | Regression-adjusted | Medium |
| **RAPM** | Ridge regression regularized | **High** |
| **BPM** | Box score estimated RAPM | Good proxy |
| **EPM** | RAPM + statistical prior | **Best** |

**Key:** RAPM explains actual impact better than any box score stat.

**Sources:**
- [Basketball-Reference BPM](https://www.basketball-reference.com/about/bpm2.html)
- [RAPM Explained](https://www.nbastuffer.com/analytics101/regularized-adjusted-plus-minus-rapm/)

### 4.2 Bayesian Performance Rating (BPR) - EvanMiya

```
BPR = Points per 100 possessions above average opponent
      if player were surrounded by 9 average players
```

**Source:** [EvanMiya CBB Analytics](https://evanmiya.com/)

### 4.3 Player Injury Impact

**Research finding:** Star player absence = **4-7 point spread swing**

Factors to model:
- Minutes played by injured player
- Usage rate
- RAPM/BPR contribution
- Depth chart quality (bench strength)

**Source:** [PointSpreads Injury Reports](https://www.pointspreads.com/ncaab/injuries/)

---

## 5. HOME COURT ADVANTAGE FACTORS

### 5.1 Quantified HCA (Average: 3.0-3.5 points)

| Factor | Impact | Source |
|--------|--------|--------|
| **Crowd density** | Primary driver | Fan support |
| **Travel distance** | -0.6 to -1.2 pts | Cross-country trips |
| **Altitude** | +0.5 to +1.0 pts | Denver, Utah venues |
| **Referee bias** | Significant | Foul calls favor home |
| **Time zone changes** | -0.3 pts/zone | Jet lag effect |
| **Conference** | Varies by 2 pts | Compact vs spread conferences |

### 5.2 Conference-Specific HCA

| Conference | Estimated HCA |
|------------|---------------|
| ACC | 4.2 pts |
| Big Ten | 4.0 pts |
| SEC | 3.8 pts |
| Big 12 | 3.5 pts |
| SoCon | 2.5 pts (short travel) |
| Blue Bloods (Kansas, Duke) | 6-7+ pts |

**Sources:**
- [Sport Journal - Home Court Advantage](https://thesportjournal.org/article/the-home-court-advantage-evidence-from-mens-college-basketball/)
- [DRatings HCA](https://www.dratings.com/home-court-advantage-in-college-basketball/)

### 5.3 COVID Research Insight

**Key finding:** Home advantage PERSISTED without fans during COVID, suggesting **venue familiarity** matters independent of crowd noise.

---

## 6. FATIGUE AND SCHEDULING

### 6.1 Rest Days Impact

| Rest Days | Impact |
|-----------|--------|
| 0 (Back-to-back) | **-2.5 to -3.0 pts** |
| 1 day | -1.0 pts |
| 2 days | Baseline (0) |
| 3+ days | +0.3 to +0.5 pts |

**Source:** [PubMed - NBA Back-to-backs](https://pubmed.ncbi.nlm.nih.gov/32172667/)

### 6.2 Schedule Congestion

NCAA Division I plays ~19 conference games in 8 weeks. Research shows:
- **Neuromuscular performance decreases** over conference season
- **Shooting efficiency drops** with schedule density
- Physical recovery between games affects outcomes

**Source:** [PMC - NCAA Division I Fatigue](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739638/)

### 6.3 Minutes-Based Fatigue

Recent research suggests **minutes played** matters more than rest days:
- High minutes in previous game → decreased +/-
- Sustained high usage → compounding fatigue effect

**Source:** [ArXiv - Modeling Player Fatigue](https://arxiv.org/pdf/2112.14649)

---

## 7. MOMENTUM AND HOT HAND

### 7.1 Hot Hand is REAL (2017+ Research)

The original 1985 study was **statistically flawed**. Modern research shows:

| Streak Length | Hot Hand Effect |
|---------------|-----------------|
| 2 makes | +2.71% FG% |
| 3 makes | +4.42% FG% |
| 4 makes | +5.81% FG% |

**Key caveat:** Effect only works for shots from SAME LOCATION.

**Sources:**
- [Scientific American - Hot Hand Vindicated](https://www.scientificamerican.com/article/momentum-isnt-magic-vindicating-the-hot-hand-with-the-mathematics-of-streaks/)
- [Springer - Bayesian Hidden Markov Hot Hand](https://link.springer.com/article/10.1007/s00180-024-01560-8)

### 7.2 Team Momentum

While individual hot hand exists, **team momentum** is harder to prove statistically. However:
- 5-game winning/losing streaks DO affect spread performance
- Recommend: **Recency-weighted efficiency** (Torvik approach)

---

## 8. REFEREE FACTORS

### 8.1 Home Team Foul Advantage

**Anderson & Pierce (2009):** Significant bias toward calling MORE fouls on visiting team in NCAA games.

**Key factors:**
- Crowd size amplifies bias
- Referee assignment patterns
- Sequential bias (evening out calls)

### 8.2 COVID Natural Experiment

Without crowds, referee bias toward home team **decreased significantly**.

**Sources:**
- [ResearchGate - NCAA Foul Calling](https://www.researchgate.net/publication/270402399_New_Insights_on_the_Tendency_of_NCAA_Basketball_Officials_to_Even_Out_Foul_Calls)
- [PMC - Crowd Impact on Referee Bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC9420652/)

---

## 9. STRENGTH OF SCHEDULE

### 9.1 NET Quadrant System (NCAA Official)

| Quadrant | Home | Neutral | Away |
|----------|------|---------|------|
| Q1 | NET 1-30 | 1-50 | 1-75 |
| Q2 | 31-75 | 51-100 | 76-135 |
| Q3 | 76-160 | 101-200 | 136-240 |
| Q4 | 161-353 | 201-353 | 241-353 |

**Source:** [NCAA.com NET Rankings](https://www.ncaa.com/news/basketball-men/article/2022-12-05/college-basketballs-net-rankings-explained)

### 9.2 Wins Above Bubble (WAB) - NEW 2025 Metric

```
WAB = Actual Wins - Expected Wins for Average Bubble Team vs Your Schedule
```

**Source:** [NCAA Selection Criteria](https://www.ncaa.org/news/2025/3/5/breaking-down-the-ncaa-division-i-mens-and-womens-basketball-committees-selection-criteria.aspx)

### 9.3 Recency-Weighted SOS (Torvik Approach)

```python
def recency_weight(days_ago):
    if days_ago <= 40:
        return 1.0
    elif days_ago <= 80:
        return 1.0 - 0.01 * (days_ago - 40)
    else:
        return 0.6  # Minimum weight
```

**Source:** [Torvik Ratings Guide](https://www.oddsshark.com/ncaab/what-are-torvik-ratings)

---

## 10. LINEUP AND ROTATION ANALYSIS

### 10.1 Five-Man Unit Efficiency

Track net rating for each 5-man combination:

```
Net Rating = (Points Scored - Points Allowed) / Possessions × 100
```

**Caution:** Most 5-man lineups have <30 possessions → high variance!

### 10.2 Lineup RAPM

Adjusted plus-minus for lineup combinations, accounting for opponent strength.

**Source:** [EvanMiya Lineup Metrics](https://evanmiya.com/)

---

## 11. TEMPO AND PACE MODELING

### 11.1 Expected Game Possessions

```
Expected_Poss = HarmonicMean(Home_Tempo, Away_Tempo) × Context_Factor

HarmonicMean = 2 × T1 × T2 / (T1 + T2)
```

### 11.2 Style Clash Factor

When tempos differ significantly:
- **Fast team vs Slow team:** High variance game
- One team forced to play uncomfortable pace
- Model should increase uncertainty

**Source:** [Maddux Sports Tempo Guide](https://www.madduxsports.com/library/cbb-handicapping/tempopace-and-offensivedefensive-efficiency-explained.html)

---

## 12. EARLY SEASON ADJUSTMENTS

### 12.1 Sample Size Stabilization

| Metric | Games to Stabilize |
|--------|-------------------|
| Win% | 15-20 games |
| AdjEM | 12-15 games |
| 3PT% | 25+ games (high variance) |
| FT% | 5-10 games |
| Tempo | 8-10 games |

### 12.2 Preseason Prior Integration

Use preseason AP poll + returning production as Bayesian prior:
- Weight: High early season, decays to zero by game 15
- AP poll is a "wisdom of crowds" predictor

**Source:** [The Power Rank - CBB Analytics Guide](https://thepowerrank.com/cbb-analytics/)

---

## 13. CONFERENCE STRENGTH

### 2025-26 Conference Power Rankings

1. **Big 12** - Most balanced top-to-bottom
2. **Big Ten** - Strong top, weak bottom
3. **SEC** - Deep but fewer elite teams
4. **ACC** - Recovering, high variance

**Key insight:** Inter-conference games early season are harder to predict due to unknown relative strength.

**Source:** [ESPN Conference Rankings](https://www.espn.com/mens-college-basketball/story/_/id/47348031/reranking-mens-basketball-power-conferences-big-ten-big-12-acc-sec-big-east)

---

## 14. CLOSING LINE VALUE (CLV)

### 14.1 Why CLV Matters

CLV is the **best predictor of long-term profitability**:
- Positive CLV = you're finding +EV bets
- Professionals track CLV over win rate

### 14.2 College Basketball CLV Caveats

**Early season:** Markets are INEFFICIENT
- Sharp money moves lines too much
- CLV less meaningful in November

**Conference season:** Markets more efficient
- CLV becomes reliable signal

**Source:** [Unabated - Precise CLV](https://unabated.com/articles/getting-precise-about-closing-line-value)

---

## 15. DATA SOURCES (FREE)

### 15.1 Primary APIs

| Source | Data Available | Access |
|--------|----------------|--------|
| **hoopR** (R) | Schedules, box scores, PBP | Free |
| **sportsdataverse-py** | Same as hoopR | Free |
| **toRvik** (R) | Torvik ratings | Free |
| **ncaa-api** | Live scores, standings | Free |

### 15.2 Analytics Platforms

| Platform | Subscription | Key Metrics |
|----------|--------------|-------------|
| KenPom | $24.95/year | AdjO, AdjD, AdjEM |
| Bart Torvik | Free | T-Rank, PRPG!, Barthag |
| EvanMiya | Free tier | BPR, Lineup data |
| ShotQuality | Paid | xFG%, shot location |

**Sources:**
- [hoopR Documentation](https://hoopr.sportsdataverse.org/)
- [sportsdataverse-py](https://sportsdataverse-py.sportsdataverse.org/)

---

## 16. IMPLEMENTATION CHECKLIST

### Tier 1: Essential (Currently Missing)

- [ ] **Four Factors** integration (eFG%, TOV%, ORB%, FTR)
- [ ] **Heteroscedastic variance** (σ varies by game type)
- [ ] **Conference-specific HCA**
- [ ] **Recency-weighted efficiency**
- [ ] **Player injury adjustments**

### Tier 2: High Impact

- [ ] **Travel distance calculation**
- [ ] **Rest days/back-to-back adjustments**
- [ ] **3PT regression to mean weighting**
- [ ] **Lineup strength when starters out**

### Tier 3: Advanced

- [ ] **Shot quality integration (if data available)**
- [ ] **Momentum/streak factors**
- [ ] **Referee assignment patterns**
- [ ] **Minutes-based fatigue modeling**

---

## 17. MATHEMATICAL FORMULATION SUMMARY

### Complete Game Prediction Model

```
E[Spread] = (Home_AdjEM - Away_AdjEM)
           + HCA_effect(home_team, conference, is_neutral)
           + Rest_adjustment(home_rest, away_rest)
           + Travel_adjustment(distance)
           + Injury_adjustment(missing_players)
           + Tempo_interaction(home_tempo, away_tempo)

σ[Spread] = σ_base
           + 0.1 × |E[Spread]|  # More variance in blowouts
           + σ_early_season     # Higher in Nov/Dec
           + σ_style_clash      # When tempos differ

E[Total] = Expected_Possessions × (Home_PPP + Away_PPP)

Win_Prob = Φ(E[Spread] / σ[Spread])  # Normal CDF
```

### Bayesian Hierarchical Structure

```
# Global priors
μ_off ~ Normal(100, 5)
σ_off ~ HalfNormal(10)

# Team abilities (partial pooling)
off_ability[team] ~ Normal(μ_off, σ_off)
def_ability[team] ~ Normal(μ_def, σ_def)

# Conference HCA
hca[conf] ~ Normal(3.5, 1.0)

# Game likelihood
margin ~ Normal(expected_margin, σ_game)
```

---

## REFERENCES

1. [KenPom Ratings](https://kenpom.com/)
2. [Bart Torvik T-Rank](https://barttorvik.com/)
3. [EvanMiya CBB Analytics](https://evanmiya.com/)
4. [Dean Oliver's Four Factors](https://arxiv.org/abs/2305.13032)
5. [Basketball-Reference](https://www.basketball-reference.com/)
6. [NCAA NET Rankings](https://www.ncaa.com/news/basketball-men/article/2022-12-05/college-basketballs-net-rankings-explained)
7. [ShotQuality](https://shotquality.com/)
8. [The Power Rank Analytics Guide](https://thepowerrank.com/cbb-analytics/)
9. [sportsdataverse](https://sportsdataverse.org/)
