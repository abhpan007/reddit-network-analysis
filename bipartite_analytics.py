#!/usr/bin/env python3
"""
Quick bipartite user-thread analytics. Run after the main pipeline.

  source .venv/bin/activate   # or your venv
  python bipartite_analytics.py

Reads existing *_user_thread.graphml (or .graphml.xml) and writes:
  - outputs/tables/bipartite_summary.csv   (n_users, n_threads, degree means/p90)
  - outputs/tables/bipartite_top_users.csv   (top users by threads participated in)
  - outputs/tables/bipartite_top_threads.csv (top threads by number of commenters)

Use these if judges ask about the bipartite graph.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
TABLES = Path("outputs/tables")
SUBREDDITS = ["hingeapp", "tinder"]


def load_bipartite(subreddit: str) -> nx.Graph | None:
    for name in [f"{subreddit}_user_thread.graphml", f"{subreddit}_user_thread.graphml.xml"]:
        path = PROCESSED / name
        if path.exists():
            return nx.read_graphml(path)
    return None


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_top_users = []
    all_top_threads = []

    for sub in SUBREDDITS:
        G = load_bipartite(sub)
        if G is None:
            print(f"[WARN] No bipartite graph for {sub}, skipping.")
            continue

        # Node sets by ID prefix (graphml stores "user::name" and "post::id")
        user_nodes = [n for n in G.nodes() if str(n).startswith("user::")]
        post_nodes = [n for n in G.nodes() if str(n).startswith("post::")]
        n_users = len(user_nodes)
        n_posts = len(post_nodes)
        n_edges = G.number_of_edges()

        # Degrees: users = threads they participated in, posts = commenters
        user_degrees = [G.degree(u) for u in user_nodes]
        post_degrees = [G.degree(p) for p in post_nodes]
        udeg = np.array(user_degrees) if user_degrees else np.array([0])
        pdeg = np.array(post_degrees) if post_degrees else np.array([0])

        summary_rows.append({
            "subreddit": sub,
            "n_users": n_users,
            "n_threads": n_posts,
            "n_edges": n_edges,
            "user_degree_mean": float(np.mean(udeg)),
            "user_degree_median": float(np.median(udeg)),
            "user_degree_p90": float(np.percentile(udeg, 90)) if len(udeg) else 0,
            "thread_degree_mean": float(np.mean(pdeg)),
            "thread_degree_median": float(np.median(pdeg)),
            "thread_degree_p90": float(np.percentile(pdeg, 90)) if len(pdeg) else 0,
        })

        # Top users by number of threads participated in
        top_u = sorted(
            [(n.replace("user::", ""), G.degree(n)) for n in user_nodes],
            key=lambda x: -x[1],
        )[:15]
        for username, deg in top_u:
            all_top_users.append({"subreddit": sub, "user": username, "threads_participated": int(deg)})

        # Top threads by number of commenters
        top_p = sorted(
            [(n.replace("post::", ""), G.degree(n)) for n in post_nodes],
            key=lambda x: -x[1],
        )[:15]
        for post_id, deg in top_p:
            all_top_threads.append({"subreddit": sub, "post_id": post_id, "n_commenters": int(deg)})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(TABLES / "bipartite_summary.csv", index=False)
        print(f"[OK] Wrote {TABLES / 'bipartite_summary.csv'}")
    if all_top_users:
        pd.DataFrame(all_top_users).to_csv(TABLES / "bipartite_top_users.csv", index=False)
        print(f"[OK] Wrote {TABLES / 'bipartite_top_users.csv'}")
    if all_top_threads:
        pd.DataFrame(all_top_threads).to_csv(TABLES / "bipartite_top_threads.csv", index=False)
        print(f"[OK] Wrote {TABLES / 'bipartite_top_threads.csv'}")
    print("[DONE] Bipartite analytics ready. Use for judge Q&A.")


if __name__ == "__main__":
    main()
