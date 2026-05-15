#!/usr/bin/env python3
"""
End-to-end Reddit network analytics pipeline for:
- r/Tinder
- r/Hinge

Outputs:
- data/raw/posts.csv
- data/raw/comments.csv
- data/processed/posts_clean.csv
- data/processed/comments_clean.csv
- data/processed/edges.csv
- outputs/tables/*.csv
- outputs/figures/*.png
- outputs/presentation_notes.md
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns

try:
    import praw
except Exception:
    praw = None

sns.set_theme(style="whitegrid")

USER_AGENT = "social-network-course-project/1.0 (by u/research_course_bot)"
BASE_URL = "https://www.reddit.com"

POST_COLUMNS = [
    "subreddit",
    "post_id",
    "permalink",
    "created_utc",
    "author",
    "title",
    "selftext",
    "post_flair_text",
    "score",
    "upvote_ratio",
    "num_comments",
    "is_self",
    "stickied",
    "locked",
    "over_18",
    "spoiler",
]

COMMENT_COLUMNS = [
    "subreddit",
    "post_id",
    "comment_id",
    "parent_id",
    "created_utc",
    "author",
    "body",
    "score",
    "depth",
    "is_submitter",
    "stickied",
]


@dataclass
class PipelineConfig:
    subreddits: list[str]
    target_posts_per_subreddit: int
    months_back: int
    include_automoderator: bool
    keep_deleted_as_node: bool
    min_pause_seconds: float


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "raw": Path("data/raw"),
        "processed": Path("data/processed"),
        "tables": Path("outputs/tables"),
        "figures": Path("outputs/figures"),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def reddit_get_json(url: str, params: dict[str, Any] | None = None, retries: int = 4) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT}
    backoff = 1.5
    for attempt in range(1, retries + 1):
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_s = max(float(retry_after), backoff * attempt)
            else:
                wait_s = backoff * attempt
            time.sleep(wait_s)
            continue
        response.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url}")


def safe_author(author: str | None, keep_deleted_as_node: bool) -> str | None:
    if author is None:
        return "deleted" if keep_deleted_as_node else None
    a = author.strip().lower()
    if a in {"[deleted]", "deleted", ""}:
        return "deleted" if keep_deleted_as_node else None
    return a


def fetch_posts_for_subreddit(subreddit: str, target_n: int, cutoff_utc: int, pause_s: float) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    sort_orders = [("new", {}), ("top", {"t": "year"}), ("hot", {})]

    for sort, extra_params in sort_orders:
        after = None
        rounds = 0
        while len(results) < target_n and rounds < 15:
            rounds += 1
            params = {"limit": 100, **extra_params}
            if after:
                params["after"] = after
            url = f"{BASE_URL}/r/{subreddit}/{sort}.json"
            try:
                payload = reddit_get_json(url, params=params)
            except requests.HTTPError as exc:
                print(f"[WARN] Failed to fetch r/{subreddit} ({sort}) page: {exc}")
                break
            children = payload.get("data", {}).get("children", [])
            if not children:
                break

            for item in children:
                d = item.get("data", {})
                created_utc = int(d.get("created_utc", 0))
                if created_utc < cutoff_utc:
                    continue

                pid = d.get("id")
                if not pid or pid in results:
                    continue

                results[pid] = {
                    "subreddit": subreddit.lower(),
                    "post_id": pid,
                    "permalink": f"https://www.reddit.com{d.get('permalink', '')}",
                    "created_utc": created_utc,
                    "author": d.get("author"),
                    "title": d.get("title"),
                    "selftext": d.get("selftext"),
                    "post_flair_text": d.get("link_flair_text"),
                    "score": d.get("score"),
                    "upvote_ratio": d.get("upvote_ratio"),
                    "num_comments": d.get("num_comments"),
                    "is_self": d.get("is_self"),
                    "stickied": d.get("stickied"),
                    "locked": d.get("locked"),
                    "over_18": d.get("over_18"),
                    "spoiler": d.get("spoiler"),
                }

                if len(results) >= target_n:
                    break

            after = payload.get("data", {}).get("after")
            if not after:
                break
            time.sleep(pause_s)

        if len(results) >= target_n:
            break

    return list(results.values())


def init_praw_client() -> Any | None:
    if praw is None:
        return None

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", USER_AGENT)
    if not client_id or not client_secret:
        return None

    try:
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
    except Exception:
        return None


def fetch_posts_for_subreddit_praw(
    reddit: Any, subreddit: str, target_n: int, cutoff_utc: int, pause_s: float
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    sr = reddit.subreddit(subreddit)
    listing_fns = [
        ("new", sr.new),
        ("top", lambda **kwargs: sr.top(time_filter="year", **kwargs)),
        ("hot", sr.hot),
    ]

    for _, listing_fn in listing_fns:
        if len(results) >= target_n:
            break
        try:
            for sub in listing_fn(limit=min(target_n * 3, 1000)):
                created_utc = int(getattr(sub, "created_utc", 0))
                if created_utc < cutoff_utc:
                    continue
                pid = getattr(sub, "id", None)
                if not pid or pid in results:
                    continue

                results[pid] = {
                    "subreddit": subreddit.lower(),
                    "post_id": pid,
                    "permalink": f"https://www.reddit.com{sub.permalink}",
                    "created_utc": created_utc,
                    "author": str(sub.author) if sub.author else None,
                    "title": sub.title,
                    "selftext": sub.selftext,
                    "post_flair_text": sub.link_flair_text,
                    "score": sub.score,
                    "upvote_ratio": getattr(sub, "upvote_ratio", None),
                    "num_comments": sub.num_comments,
                    "is_self": sub.is_self,
                    "stickied": sub.stickied,
                    "locked": sub.locked,
                    "over_18": sub.over_18,
                    "spoiler": sub.spoiler,
                }
                if len(results) >= target_n:
                    break
            time.sleep(pause_s)
        except Exception as exc:
            print(f"[WARN] PRAW listing failed for r/{subreddit}: {exc}")
            continue

    return list(results.values())


def flatten_comments_tree(
    children: list[dict[str, Any]],
    subreddit: str,
    post_id: str,
    depth: int,
    out_rows: list[dict[str, Any]],
) -> None:
    for child in children:
        kind = child.get("kind")
        data = child.get("data", {})
        if kind == "more":
            # Public JSON endpoint may include unresolved "more" placeholders.
            continue
        if kind != "t1":
            continue

        out_rows.append(
            {
                "subreddit": subreddit.lower(),
                "post_id": post_id,
                "comment_id": data.get("id"),
                "parent_id": data.get("parent_id"),
                "created_utc": data.get("created_utc"),
                "author": data.get("author"),
                "body": data.get("body"),
                "score": data.get("score"),
                "depth": depth,
                "is_submitter": data.get("is_submitter"),
                "stickied": data.get("stickied"),
            }
        )

        replies = data.get("replies")
        if isinstance(replies, dict):
            nested = replies.get("data", {}).get("children", [])
            flatten_comments_tree(nested, subreddit, post_id, depth + 1, out_rows)


def fetch_comments_for_post(subreddit: str, post_id: str, pause_s: float) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/comments/{post_id}.json"
    try:
        payload = reddit_get_json(url, params={"limit": 500, "sort": "best"})
    except requests.HTTPError as exc:
        print(f"[WARN] Failed comments for {subreddit}/{post_id}: {exc}")
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list) and len(payload) >= 2:
        comment_listing = payload[1]
        children = comment_listing.get("data", {}).get("children", [])
        flatten_comments_tree(children, subreddit, post_id, 0, rows)
    time.sleep(pause_s)
    return rows


def fetch_comments_for_post_praw(reddit: Any, subreddit: str, post_id: str, pause_s: float) -> list[dict[str, Any]]:
    try:
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=None)
        comments = submission.comments.list()
    except Exception as exc:
        print(f"[WARN] PRAW comments failed for {subreddit}/{post_id}: {exc}")
        return []

    rows: list[dict[str, Any]] = []
    for c in comments:
        rows.append(
            {
                "subreddit": subreddit.lower(),
                "post_id": post_id,
                "comment_id": c.id,
                "parent_id": c.parent_id,
                "created_utc": c.created_utc,
                "author": str(c.author) if c.author else None,
                "body": c.body,
                "score": c.score,
                "depth": c.depth,
                "is_submitter": c.is_submitter,
                "stickied": c.stickied,
            }
        )
    time.sleep(pause_s)
    return rows


def clean_posts(posts_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if posts_df.empty:
        return pd.DataFrame(columns=POST_COLUMNS)
    for col in POST_COLUMNS:
        if col not in posts_df.columns:
            posts_df[col] = np.nan
    df = posts_df.drop_duplicates(subset=["post_id"]).copy()
    df["author"] = df["author"].apply(lambda x: safe_author(x, cfg.keep_deleted_as_node))
    if not cfg.include_automoderator:
        df = df[df["author"] != "automoderator"]
    df = df[df["author"].notna()]
    return df


def clean_comments(comments_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if comments_df.empty:
        return pd.DataFrame(columns=COMMENT_COLUMNS)
    for col in COMMENT_COLUMNS:
        if col not in comments_df.columns:
            comments_df[col] = np.nan
    df = comments_df.drop_duplicates(subset=["comment_id"]).copy()
    df["author"] = df["author"].apply(lambda x: safe_author(x, cfg.keep_deleted_as_node))
    if not cfg.include_automoderator:
        df = df[df["author"] != "automoderator"]
    df = df[df["author"].notna()]
    df["body"] = df["body"].fillna("").astype(str)
    return df


def build_edges(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
    post_author = posts_df.set_index("post_id")["author"].to_dict()
    comment_author = comments_df.set_index("comment_id")["author"].to_dict()

    rows: list[dict[str, Any]] = []
    for _, row in comments_df.iterrows():
        parent = str(row["parent_id"] or "")
        src = row["author"]
        subreddit = row["subreddit"]
        post_id = row["post_id"]
        if not src:
            continue

        if parent.startswith("t1_"):
            parent_comment_id = parent.replace("t1_", "")
            dst = comment_author.get(parent_comment_id)
            if dst and dst != src:
                rows.append(
                    {
                        "subreddit": subreddit,
                        "src_user": src,
                        "dst_user": dst,
                        "edge_type": "reply_to_user",
                        "post_id": post_id,
                    }
                )
        elif parent.startswith("t3_"):
            dst = post_author.get(post_id)
            if dst and dst != src:
                rows.append(
                    {
                        "subreddit": subreddit,
                        "src_user": src,
                        "dst_user": dst,
                        "edge_type": "reply_to_op",
                        "post_id": post_id,
                    }
                )

    edges = pd.DataFrame(rows)
    if edges.empty:
        return pd.DataFrame(columns=["subreddit", "src_user", "dst_user", "edge_type", "weight"])

    agg = (
        edges.groupby(["subreddit", "src_user", "dst_user", "edge_type"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )
    return agg


def build_user_graph(edges_df: pd.DataFrame, subreddit: str) -> nx.DiGraph:
    sdf = edges_df[edges_df["subreddit"] == subreddit]
    g = nx.DiGraph()
    for _, row in sdf.iterrows():
        g.add_edge(row["src_user"], row["dst_user"], weight=float(row["weight"]))
    return g


def build_user_thread_graph(comments_df: pd.DataFrame, subreddit: str) -> nx.Graph:
    sdf = comments_df[comments_df["subreddit"] == subreddit]
    counts = sdf.groupby(["author", "post_id"], as_index=False).size().rename(columns={"size": "weight"})
    b = nx.Graph()
    for _, row in counts.iterrows():
        user = f"user::{row['author']}"
        thread = f"post::{row['post_id']}"
        b.add_node(user, bipartite="user")
        b.add_node(thread, bipartite="post")
        b.add_edge(user, thread, weight=float(row["weight"]))
    return b


def directed_density(g: nx.DiGraph) -> float:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n <= 1:
        return 0.0
    return m / (n * (n - 1))


def giant_component_subgraph_directed(g: nx.DiGraph) -> nx.DiGraph:
    if g.number_of_nodes() == 0:
        return g.copy()
    components = list(nx.weakly_connected_components(g))
    giant = max(components, key=len)
    return g.subgraph(giant).copy()


def compute_network_metrics(g: nx.DiGraph) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n = g.number_of_nodes()
    m = g.number_of_edges()
    metrics["nodes"] = float(n)
    metrics["edges"] = float(m)

    weak_components = list(nx.weakly_connected_components(g)) if n else []
    metrics["weak_components"] = float(len(weak_components))
    giant_size = len(max(weak_components, key=len)) if weak_components else 0
    metrics["giant_component_pct"] = float((giant_size / n) * 100.0) if n else 0.0
    metrics["directed_density"] = directed_density(g)

    in_degrees = np.array([d for _, d in g.in_degree()], dtype=float) if n else np.array([0.0])
    out_degrees = np.array([d for _, d in g.out_degree()], dtype=float) if n else np.array([0.0])
    metrics["in_degree_mean"] = float(np.mean(in_degrees))
    metrics["out_degree_mean"] = float(np.mean(out_degrees))
    metrics["in_degree_p90"] = float(np.percentile(in_degrees, 90))
    metrics["out_degree_p90"] = float(np.percentile(out_degrees, 90))

    und = g.to_undirected()
    metrics["global_clustering"] = float(nx.transitivity(und)) if und.number_of_nodes() > 2 else 0.0
    metrics["avg_local_clustering"] = float(nx.average_clustering(und)) if und.number_of_nodes() > 2 else 0.0

    giant = giant_component_subgraph_directed(g).to_undirected()
    if giant.number_of_nodes() > 1:
        try:
            metrics["avg_shortest_path"] = float(nx.average_shortest_path_length(giant))
        except Exception:
            metrics["avg_shortest_path"] = float("nan")
        try:
            metrics["diameter"] = float(nx.diameter(giant))
        except Exception:
            metrics["diameter"] = float("nan")
    else:
        metrics["avg_shortest_path"] = float("nan")
        metrics["diameter"] = float("nan")
    return metrics


def top_centralities(g: nx.DiGraph, top_n: int = 20) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "metric", "score", "rank"])

    in_deg = nx.in_degree_centrality(g)
    out_deg = nx.out_degree_centrality(g)
    between = nx.betweenness_centrality(g, k=min(200, g.number_of_nodes()), normalized=True, seed=42)

    und = g.to_undirected()
    if und.number_of_nodes() > 1:
        close = nx.closeness_centrality(und)
        try:
            pagerank = nx.pagerank(g, weight="weight")
        except Exception:
            # Fallback when SciPy-backed implementation is unavailable.
            pagerank = nx.eigenvector_centrality_numpy(und, weight="weight")
    else:
        close = {n: 0.0 for n in g.nodes()}
        pagerank = {n: 0.0 for n in g.nodes()}

    metric_map = {
        "in_degree_centrality": in_deg,
        "out_degree_centrality": out_deg,
        "betweenness_centrality": between,
        "closeness_centrality": close,
        "pagerank": pagerank,
    }

    rows: list[dict[str, Any]] = []
    for metric_name, score_map in metric_map.items():
        top_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for rank, (user, score) in enumerate(top_items, start=1):
            rows.append({"user": user, "metric": metric_name, "score": float(score), "rank": rank})
    return pd.DataFrame(rows)


def detect_communities(g: nx.DiGraph) -> tuple[pd.DataFrame, float]:
    und = g.to_undirected()
    if und.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "community"]), float("nan")
    if und.number_of_edges() == 0:
        only = list(und.nodes())
        return pd.DataFrame({"user": only, "community": [0] * len(only)}), 0.0

    try:
        communities = nx.community.louvain_communities(und, weight="weight", seed=42)
    except Exception:
        communities = nx.community.greedy_modularity_communities(und, weight="weight")

    comm_rows = []
    for idx, members in enumerate(sorted(communities, key=len, reverse=True)):
        for u in members:
            comm_rows.append({"user": u, "community": idx})

    modularity = nx.community.modularity(und, communities, weight="weight")
    return pd.DataFrame(comm_rows), float(modularity)


def derive_user_attributes(posts_df: pd.DataFrame, comments_df: pd.DataFrame, subreddit: str) -> pd.DataFrame:
    p = posts_df[posts_df["subreddit"] == subreddit]
    c = comments_df[comments_df["subreddit"] == subreddit]

    post_counts = p.groupby("author").size().rename("posts_authored")
    comment_counts = c.groupby("author").size().rename("comments_authored")

    users = pd.Index(sorted(set(post_counts.index).union(set(comment_counts.index))))
    df = pd.DataFrame(index=users)
    df = df.join(post_counts, how="left").join(comment_counts, how="left").fillna(0.0)
    df["posts_authored"] = df["posts_authored"].astype(int)
    df["comments_authored"] = df["comments_authored"].astype(int)

    df["role"] = np.where(df["posts_authored"] > df["comments_authored"], "op_heavy", "commenter_heavy")
    median_comments = float(df["comments_authored"].median()) if len(df) else 0.0
    df["engagement_tier"] = np.where(df["comments_authored"] >= median_comments, "high", "low")
    df = df.reset_index(names=["user"])
    return df


def e_i_index(g: nx.Graph, attr_map: dict[str, str]) -> float:
    internal = 0
    external = 0
    for u, v in g.edges():
        au = attr_map.get(u)
        av = attr_map.get(v)
        if au is None or av is None:
            continue
        if au == av:
            internal += 1
        else:
            external += 1
    denom = internal + external
    if denom == 0:
        return float("nan")
    return (external - internal) / denom


def compute_homophily(g: nx.DiGraph, attrs: pd.DataFrame) -> pd.DataFrame:
    und = g.to_undirected()
    if und.number_of_nodes() == 0:
        return pd.DataFrame(columns=["attribute", "assortativity", "e_i_index"])

    attr_df = attrs.set_index("user")
    rows = []
    for attr in ["role", "engagement_tier"]:
        mapping = attr_df[attr].to_dict()
        nx.set_node_attributes(und, mapping, attr)
        try:
            assort = nx.attribute_assortativity_coefficient(und, attr)
        except Exception:
            assort = float("nan")
        ei = e_i_index(und, mapping)
        rows.append({"attribute": attr, "assortativity": float(assort), "e_i_index": float(ei)})
    return pd.DataFrame(rows)


def plot_degree_distribution(g: nx.DiGraph, subreddit: str, fig_dir: Path) -> None:
    in_degrees = [d for _, d in g.in_degree()]
    out_degrees = [d for _, d in g.out_degree()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(in_degrees, bins=30, ax=axes[0], kde=False)
    axes[0].set_title(f"{subreddit}: In-degree distribution")
    axes[0].set_xlabel("in-degree")
    sns.histplot(out_degrees, bins=30, ax=axes[1], kde=False)
    axes[1].set_title(f"{subreddit}: Out-degree distribution")
    axes[1].set_xlabel("out-degree")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{subreddit}_degree_distribution.png", dpi=220)
    plt.close(fig)


def plot_component_sizes(g: nx.DiGraph, subreddit: str, fig_dir: Path) -> None:
    comps = [len(c) for c in nx.weakly_connected_components(g)] if g.number_of_nodes() else []
    if not comps:
        comps = [0]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=list(range(1, len(comps) + 1)), y=sorted(comps, reverse=True), ax=ax, color="#4C72B0")
    ax.set_title(f"{subreddit}: weakly connected component sizes")
    ax.set_xlabel("component rank")
    ax.set_ylabel("size")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{subreddit}_component_sizes.png", dpi=220)
    plt.close(fig)


def plot_community_graph(g: nx.DiGraph, comm_df: pd.DataFrame, subreddit: str, fig_dir: Path) -> None:
    und = giant_component_subgraph_directed(g).to_undirected()
    if und.number_of_nodes() == 0:
        return
    cmap = {}
    for _, row in comm_df.iterrows():
        cmap[row["user"]] = row["community"]
    colors = [cmap.get(node, -1) for node in und.nodes()]
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(und, seed=42, k=1 / math.sqrt(max(1, und.number_of_nodes())))
    nx.draw_networkx_nodes(und, pos, node_size=30, node_color=colors, cmap="tab20", ax=ax, alpha=0.9)
    nx.draw_networkx_edges(und, pos, width=0.2, alpha=0.25, ax=ax)
    ax.set_title(f"{subreddit}: community-colored giant component")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{subreddit}_communities.png", dpi=240)
    plt.close(fig)


def plot_ego_nets(g: nx.DiGraph, top_users: list[str], subreddit: str, fig_dir: Path) -> None:
    und = g.to_undirected()
    for user in top_users[:2]:
        if user not in und:
            continue
        ego = nx.ego_graph(und, user, radius=1)
        if ego.number_of_nodes() < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        pos = nx.spring_layout(ego, seed=42)
        colors = ["#DD8452" if n == user else "#4C72B0" for n in ego.nodes()]
        nx.draw_networkx_nodes(ego, pos, node_size=120, node_color=colors, ax=ax)
        nx.draw_networkx_edges(ego, pos, width=0.8, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(ego, pos, labels={user: user}, font_size=8, ax=ax)
        ax.set_title(f"{subreddit}: ego network of {user}")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{subreddit}_ego_{user}.png", dpi=240)
        plt.close(fig)


def write_presentation_notes(
    metrics_df: pd.DataFrame,
    homophily_map: dict[str, pd.DataFrame],
    community_summary: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    lines = [
        "# Reddit Dating App Network Analytics",
        "",
        "## Motivation",
        "- Compare interaction structures in r/Tinder vs r/Hinge.",
        "- Identify central users, communities, and mixing patterns.",
        "",
        "## Method",
        "- Data: public Reddit JSON endpoints (posts + comment trees where available).",
        "- Networks: directed user-user reply graph + user-thread bipartite graph.",
        "- Metrics: density, clustering, path metrics, centralities, communities, homophily.",
        "",
        "## Key Results (drop directly into slides)",
    ]

    for _, row in metrics_df.iterrows():
        sub = row["subreddit"]
        lines += [
            "",
            f"### {sub}",
            f"- Nodes: {int(row['nodes'])}, Edges: {int(row['edges'])}",
            f"- Density: {row['directed_density']:.4f}",
            f"- Giant component: {row['giant_component_pct']:.1f}%",
            f"- Global clustering: {row['global_clustering']:.4f}",
            f"- Avg shortest path: {row['avg_shortest_path']:.3f}",
        ]
        comm = community_summary.get(sub, {})
        if comm:
            lines.append(
                f"- Communities: {comm.get('num_communities', 0)} (modularity={comm.get('modularity', float('nan')):.4f})"
            )
        hdf = homophily_map.get(sub)
        if hdf is not None and not hdf.empty:
            for _, h in hdf.iterrows():
                lines.append(
                    f"- Homophily `{h['attribute']}`: assortativity={h['assortativity']:.4f}, E-I={h['e_i_index']:.4f}"
                )

    lines += [
        "",
        "## Figures to Include",
        "- Degree distribution per subreddit",
        "- Component size bar charts",
        "- Community-colored giant components",
        "- Two ego-networks per subreddit",
        "",
        "## Limitations + Ethics",
        "- Public-only data and aggregate reporting only.",
        "- JSON endpoint does not fully resolve all `more comments`; API-auth scraping is better for full trees.",
        "- Usernames are pseudonymous and may not represent stable identities.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reddit dating-app network analysis pipeline.")
    parser.add_argument("--subreddits", nargs="+", default=["Tinder", "Hinge"])
    parser.add_argument("--target-posts", type=int, default=400)
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument("--include-automoderator", action="store_true")
    parser.add_argument("--keep-deleted-as-node", action="store_true")
    parser.add_argument("--pause", type=float, default=0.75, help="seconds between requests")
    parser.add_argument(
        "--source",
        choices=["auto", "praw", "public_json"],
        default="auto",
        help="Use authenticated PRAW when credentials are available.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        subreddits=[s.lower() for s in args.subreddits],
        target_posts_per_subreddit=args.target_posts,
        months_back=args.months_back,
        include_automoderator=args.include_automoderator,
        keep_deleted_as_node=args.keep_deleted_as_node,
        min_pause_seconds=args.pause,
    )

    paths = ensure_dirs()
    cutoff = int((datetime.now(timezone.utc) - timedelta(days=30 * cfg.months_back)).timestamp())
    reddit = init_praw_client()
    if args.source == "praw" and reddit is None:
        raise RuntimeError(
            "PRAW requested but missing credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )
    use_praw = reddit is not None and args.source in {"auto", "praw"}
    print(f"[INFO] Data source: {'praw' if use_praw else 'public_json'}")

    all_posts: list[dict[str, Any]] = []
    all_comments: list[dict[str, Any]] = []

    for sub in cfg.subreddits:
        print(f"[INFO] Scraping posts for r/{sub} ...")
        if use_praw:
            posts = fetch_posts_for_subreddit_praw(
                reddit=reddit,
                subreddit=sub,
                target_n=cfg.target_posts_per_subreddit,
                cutoff_utc=cutoff,
                pause_s=cfg.min_pause_seconds,
            )
        else:
            posts = fetch_posts_for_subreddit(
                subreddit=sub,
                target_n=cfg.target_posts_per_subreddit,
                cutoff_utc=cutoff,
                pause_s=cfg.min_pause_seconds,
            )
        print(f"[INFO] r/{sub}: collected {len(posts)} posts")
        all_posts.extend(posts)

        for idx, post in enumerate(posts, start=1):
            if use_praw:
                crows = fetch_comments_for_post_praw(reddit, sub, post["post_id"], pause_s=cfg.min_pause_seconds)
            else:
                crows = fetch_comments_for_post(sub, post["post_id"], pause_s=cfg.min_pause_seconds)
            all_comments.extend(crows)
            if idx % 25 == 0:
                print(f"[INFO] r/{sub}: comments fetched for {idx}/{len(posts)} posts")

    posts_df_raw = pd.DataFrame(all_posts, columns=POST_COLUMNS)
    comments_df_raw = pd.DataFrame(all_comments, columns=COMMENT_COLUMNS)
    posts_df_raw.to_csv(paths["raw"] / "posts.csv", index=False)
    comments_df_raw.to_csv(paths["raw"] / "comments.csv", index=False)

    posts_df = clean_posts(posts_df_raw, cfg)
    comments_df = clean_comments(comments_df_raw, cfg)
    edges_df = build_edges(posts_df, comments_df)

    posts_df.to_csv(paths["processed"] / "posts_clean.csv", index=False)
    comments_df.to_csv(paths["processed"] / "comments_clean.csv", index=False)
    edges_df.to_csv(paths["processed"] / "edges.csv", index=False)

    basic_counts = []
    metrics_rows = []
    homophily_map: dict[str, pd.DataFrame] = {}
    community_summary: dict[str, dict[str, Any]] = {}

    for sub in cfg.subreddits:
        g = build_user_graph(edges_df, sub)
        b = build_user_thread_graph(comments_df, sub)

        nx.write_graphml(g, paths["processed"] / f"{sub}_user_user.graphml")
        nx.write_graphml(b, paths["processed"] / f"{sub}_user_thread.graphml")

        counts = {
            "subreddit": sub,
            "posts": int((posts_df["subreddit"] == sub).sum()),
            "comments": int((comments_df["subreddit"] == sub).sum()),
            "unique_users": int(
                len(
                    set(posts_df.loc[posts_df["subreddit"] == sub, "author"]).union(
                        set(comments_df.loc[comments_df["subreddit"] == sub, "author"])
                    )
                )
            ),
            "edges": int((edges_df["subreddit"] == sub).sum()) if not edges_df.empty else 0,
        }
        basic_counts.append(counts)

        nmetrics = compute_network_metrics(g)
        nmetrics["subreddit"] = sub
        metrics_rows.append(nmetrics)

        tops = top_centralities(g, top_n=20)
        tops.to_csv(paths["tables"] / f"{sub}_top_centralities.csv", index=False)

        overlap = (
            tops.sort_values(["metric", "rank"])
            .groupby("metric")["user"]
            .apply(set)
            .to_dict()
            if not tops.empty
            else {}
        )
        overlap_rows = []
        metric_names = sorted(list(overlap.keys()))
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                a = metric_names[i]
                bname = metric_names[j]
                inter = len(overlap[a].intersection(overlap[bname]))
                overlap_rows.append({"metric_a": a, "metric_b": bname, "top20_overlap_count": inter})
        pd.DataFrame(overlap_rows).to_csv(paths["tables"] / f"{sub}_centrality_overlap.csv", index=False)

        comm_df, modularity = detect_communities(g)
        comm_df.to_csv(paths["tables"] / f"{sub}_community_membership.csv", index=False)
        comm_sizes = comm_df.groupby("community").size().sort_values(ascending=False)
        comm_sizes.to_csv(paths["tables"] / f"{sub}_community_sizes.csv", header=["size"])
        community_summary[sub] = {
            "num_communities": int(comm_df["community"].nunique()) if not comm_df.empty else 0,
            "modularity": modularity,
        }

        attrs = derive_user_attributes(posts_df, comments_df, sub)
        attrs.to_csv(paths["tables"] / f"{sub}_user_attributes.csv", index=False)
        hdf = compute_homophily(g, attrs)
        hdf.to_csv(paths["tables"] / f"{sub}_homophily.csv", index=False)
        homophily_map[sub] = hdf

        plot_degree_distribution(g, sub, paths["figures"])
        plot_component_sizes(g, sub, paths["figures"])
        plot_community_graph(g, comm_df, sub, paths["figures"])

        pr_users = (
            tops[tops["metric"] == "pagerank"]
            .sort_values("rank")
            .head(2)["user"]
            .tolist()
            if not tops.empty
            else []
        )
        plot_ego_nets(g, pr_users, sub, paths["figures"])

    basic_counts_df = pd.DataFrame(basic_counts)
    basic_counts_df.to_csv(paths["tables"] / "basic_counts.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[
        [
            "subreddit",
            "nodes",
            "edges",
            "weak_components",
            "giant_component_pct",
            "directed_density",
            "in_degree_mean",
            "out_degree_mean",
            "in_degree_p90",
            "out_degree_p90",
            "global_clustering",
            "avg_local_clustering",
            "avg_shortest_path",
            "diameter",
        ]
    ]
    metrics_df.to_csv(paths["tables"] / "network_metrics_comparison.csv", index=False)

    write_presentation_notes(
        metrics_df=metrics_df,
        homophily_map=homophily_map,
        community_summary=community_summary,
        out_path=Path("outputs/presentation_notes.md"),
    )

    print("[DONE] Pipeline completed.")
    print("[DONE] Check outputs/tables, outputs/figures, and outputs/presentation_notes.md")


if __name__ == "__main__":
    main()
