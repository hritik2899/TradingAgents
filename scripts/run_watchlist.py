"""Run TradingAgents sequentially for the configured watchlist."""

import json
import os
from datetime import date
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients import create_llm_client

WATCHLIST = [
    ("KEC International", "KEC.NS", "stock", ("market", "social", "news", "fundamentals")),
    ("KPIT Technologies", "KPITTECH.NS", "stock", ("market", "social", "news", "fundamentals")),
    ("SPARC", "SPARC.NS", "stock", ("market", "social", "news", "fundamentals")),
    ("Zaggle", "ZAGGLE.NS", "stock", ("market", "social", "news", "fundamentals")),
    ("Silver", "SI=F", "commodity", ("market", "social", "news")),
    ("Copper", "HG=F", "commodity", ("market", "social", "news")),
    ("Bitcoin", "BTC-USD", "crypto", ("market", "social", "news")),
]

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def build_config(output_dir: Path) -> dict:
    provider = os.getenv("TRADINGAGENTS_LLM_PROVIDER", "gemini").lower()
    default_model = DEFAULT_GEMINI_MODEL if provider in {"google", "gemini"} else "gpt-5.4-mini"
    config = dict(DEFAULT_CONFIG)
    config["llm_provider"] = provider
    config["deep_think_llm"] = os.getenv("TRADINGAGENTS_DEEP_THINK_LLM", default_model)
    config["quick_think_llm"] = os.getenv("TRADINGAGENTS_QUICK_THINK_LLM", default_model)
    if provider == "nvidia":
        config["backend_url"] = "https://integrate.api.nvidia.com/v1"
    else:
        config["backend_url"] = None
    config["results_dir"] = str(output_dir)
    config["checkpoint_enabled"] = False
    config["max_debate_rounds"] = int(os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1"))
    config["max_risk_discuss_rounds"] = int(os.getenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "1"))
    config["llm_max_retries"] = 3
    return config


def preflight_llm(config: dict) -> None:
    """Make one tiny LLM request before starting the seven-asset run."""
    provider = config["llm_provider"]
    model = config["quick_think_llm"]
    if provider in {"gemini", "google"} and not (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    ):
        raise RuntimeError("GEMINI_API_KEY is not configured in the workflow secrets")
    client = create_llm_client(
        provider=provider,
        model=model,
        base_url=config.get("backend_url"),
        api_key=os.getenv("GEMINI_API_KEY") if provider in {"gemini", "google"} else None,
    )
    response = client.get_llm().invoke("Reply with exactly: OK")
    print(f"LLM preflight passed: provider={provider}, model={model}, response={response.content!r}")


def main() -> None:
    run_date = os.getenv("TRADINGAGENTS_RUN_DATE") or date.today().isoformat()
    output_dir = Path(os.getenv("TRADINGAGENTS_RESULTS_DIR", "results/watchlist"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(output_dir)

    # Fail fast on provider/key/model problems instead of burning calls across
    # all seven assets when the LLM is unavailable.
    preflight_llm(config)

    summaries = []
    for name, ticker, asset_type, analysts in WATCHLIST:
        print(f"\n{'=' * 80}\n{name} ({ticker}) [{asset_type}] — {run_date}\n{'=' * 80}")
        try:
            graph = TradingAgentsGraph(
                selected_analysts=analysts,
                debug=False,
                config=config,
            )
            _, decision = graph.propagate(ticker, run_date, asset_type=asset_type)
            result = {
                "name": name,
                "ticker": ticker,
                "asset_type": asset_type,
                "date": run_date,
                "decision": decision,
                "status": "ok",
            }
            summaries.append(result)
            print(json.dumps(result, default=str, indent=2))
        except Exception as exc:
            result = {
                "name": name,
                "ticker": ticker,
                "asset_type": asset_type,
                "date": run_date,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            summaries.append(result)
            print(json.dumps(result, indent=2))

    summary_path = output_dir / "watchlist_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
