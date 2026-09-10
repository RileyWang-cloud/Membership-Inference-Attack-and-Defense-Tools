# Recommender Privacy Defenses

This folder includes supplementary defenses for recommender-system membership
inference experiments.

## Popularity Randomization

`PopularityRandomizationDefense` implements the paper-style Popularity
Randomization defense for recommendation lists. It first sorts items by
popularity, selects the top `Ncand` items as candidates, and randomly samples
`Nrec` items as the exposed non-member recommendations.

The default follows the paper notation:

```text
alpha_pr = Nrec / Ncand = 0.1
Scand = first Ncand items in Ssorted
Rout = random sample from Scand, where |Rout| = Nrec
```

By default, only `nonmember_recommendations` are randomized. Member
recommendations are left unchanged.

This is intended for supplementary experiments around:

- membership inference attacks against recommender systems
- membership inference attacks against sequential recommender systems

The defense accepts the same raw recommendation-list payload style used by
`Attack/biased_mia.py` and `Attack/dl_mia.py`:

```python
{
    "member_interactions": {...},
    "nonmember_interactions": {...},
    "member_recommendations": {...},
    "nonmember_recommendations": {...},
}
```

Run the demo:

```bash
python Defense/rec_privacy_defense_demo.py
```

Useful knobs:

```bash
python Defense/rec_privacy_defense_demo.py --replacement-probability 0.5
python Defense/rec_privacy_defense_demo.py --alpha-pr 0.1
python Defense/rec_privacy_defense_demo.py --candidate-pool-size 60
```

`candidate_pool_size` is an override for ablation. If omitted, each user's
candidate size is computed as `ceil(Nrec / alpha_pr)`.

## Recommendation List Shuffle

`RecommendationListShuffleDefense` implements the A.5-style defense mechanism
that shuffles the exposed recommendation list. It does not change model
parameters and does not change the recommended item set. It only randomizes the
order of each list, so rank-position information is no longer reliable for
Biased-MIA, DL-MIA, or ME-MIA variants that consume ranking signals.

The defense follows the unified defense interface:

- input: `DefenseInput(samples=...)` or `DefenseInput(signals=...)`
- output: `DefenseOutput(transformed_data=..., protected_outputs=...)`
- optional deployment output: `protected_predictor` when `target_model` is
  provided

Accepted payload format is the same as above:

```python
{
    "member_interactions": {...},
    "nonmember_interactions": {...},
    "member_recommendations": {...},
    "nonmember_recommendations": {...},
}
```

By default, both member and non-member recommendation lists are shuffled because
the deployed target model exposes shuffled rankings for every queried user.

Run the demo with this defense:

```bash
python Defense/rec_privacy_defense_demo.py --defense shuffle
python Defense/rec_privacy_defense_demo.py --defense shuffle --shuffle-probability 0.5
```
