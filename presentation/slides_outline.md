# Slide 1 — Title
- **Reddit Dating App Communities**
- Course, team members, date

# Slide 2 — Motivation + Research Questions
- How do interaction structures differ between `r/Tinder` and `r/Hinge`?
- Are there stronger communities in one subreddit?
- Is there more homophily (within-group interaction) in one subreddit?

# Slide 3 — Data + Scope
- Subreddits: `r/Tinder`, `r/Hinge`
- Time window: last 6 months
- Target size: 400 posts each (or sampled subset if API limits)
- Data units: posts, comments, reply edges
- Ethics: public data only, aggregate-level reporting

# Slide 4 — Data Pipeline
- Collection: Reddit JSON endpoints
- Cleaning: deduplication, lowercase usernames, drop/delete policy
- Edge building:
  - `reply_to_user`: commenter -> parent comment author
  - `reply_to_op`: commenter -> submission author

# Slide 5 — Network Definitions
- **Network 1 (core):** directed weighted user-user interaction graph
- **Network 2:** user-thread bipartite graph
- Node and edge semantics shown visually

# Slide 6 — Network-Level Comparison
- Table from `outputs/tables/network_metrics_comparison.csv`
- Highlight: nodes, edges, density, giant component, clustering, path length

# Slide 7 — Degree + Component Structure
- Degree distribution figures
- Component size figures
- Interpretation: centralization and fragmentation differences

# Slide 8 — Node-Level Importance
- Top-20 users by in-degree, out-degree, betweenness, closeness, PageRank
- Overlap results from centrality overlap table

# Slide 9 — Communities
- Community-colored graph for each subreddit (`hingeapp_communities.png`, `tinder_communities.png`)
- Community size distribution + modularity
- **Labels (evidence from outputs):** See `outputs/INSIGHTS_PLAN.md` Part 3. Examples:
  - **r/Hinge:** Largest = “Core advice hub”; next = “Secondary discussion clusters”; small = “Peripheral/dyadic.”
  - **r/Tinder:** Largest = “Main discussion cluster”; next = “Major subgroups”; mid = “Topic/style clusters”; small = “Peripheral.”

# Slide 10 — Homophily Results
- Assortativity and E-I index for:
  - `role` (op-heavy vs commenter-heavy)
  - `engagement_tier` (high vs low)
- Which subreddit is more echo-chamber-like?

# Slide 11 — Tinder vs Hinge Story
- **r/Hinge:** Denser, more clustered, shorter paths, generalized hubs → centralized “profile and dating advice” culture.
- **r/Tinder:** Larger, more modular, longer paths, diverse central users → multi-cluster “big tent” culture with distinct subgroups.
- Use 3–5 bullets from `outputs/INSIGHTS_PLAN.md` Part 2 (Culture Story) to tie structure → app/subreddit culture.
- Example: one subreddit is centralized around advice hubs; the other is more distributed with many niches.

# Slide 12 — Limitations + Ethics + Future Work
- API endpoint limitations (`more comments`)
- Sampling/time-window caveats
- Ethical handling and de-identification
- Future: sentiment/topic attributes for stronger homophily tests
