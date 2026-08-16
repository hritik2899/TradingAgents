"""Run TradingAgents sequentially for a fixed watchlist."""
import json
import os
from datetime import date
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

WATCHLIST = [
    ("KEC International", "KEC.NS", ("market", "social", "news", "fundamentals")),
    ("KPIT Technologies", "KPITTECH.NS", ("market", "social", "news", "fundamentals")),
    ("SPARC", "SPARC.NS", ("market", "social", "news", "fundamentals")),
    ("Zaggle", "ZAGGLE.NS", ("market", "social", "news", "fundamentals")),
    ("Silver", "SI=F", ("market", "social", "news")),
    ("Copper", "HG=F", ("market", "social", "news")),
    ("Bitcoin", "BTC-USD", ("market", "social", "news")),
]


def main():
    run_date = os.getenv("TRADINGAGENTS_RUN_DATE") or date.today().isoformat()
    output_dir = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "results/watchlist"))
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = os.getenv("TRADINGAGENTS_LLM_PROVIDER", "nvidia")
    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = provider
    config["deep_think_llm"] = os.getenv("TRADINGAGENTS_DEEP_THINK_LLM", "z-ai/glm-5.2")
    config["quick_think_llm"] = os.getenv("TRADINGAGENTS_QUICK_THINK_LLM", "z-ai/glm-5.2")
    if provider == "nvidia":
        config["backend_url"] = "https://integrate.api.nvidia.com/v1"
    config["results_dir"] = str(output_dir)
    config["checkpoint_enabled"] = False
    config["max_debate_rounds"] = int(os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1"))
    config["max_risk_discuss_rounds"] = int(os.getenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "1"))
    config["llm_max_retries"] = 3

    summaries = []
    for name, ticker, analysts in WATCHLIST:
        print(f"\n{'=' * 80}\n{name} ({ticker}) — {run_date}\n{'=' * 80}")
        try:
            graph = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=config)
            _, decision = graph.propagate(ticker, run_date)
            result = {"name": name, "ticker": ticker, "date": run_date, "decision": decision, "status": "ok"}
            summaries.append(result)
            print(json.dumps(result, default=str, indent=2))
        except Exception as exc:
            result = {"name": name, "ticker": ticker, "date": run_date, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            summaries.append(result)
            print(json.dumps(result, indent=2))

    summary_path = output_dir / "watchlist_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
