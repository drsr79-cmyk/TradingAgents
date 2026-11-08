"""TradingAgents: multi-agent LLM financial trading framework."""

__version__ = "0.7.1"

from tradingagents.safety import install_secret_filter, load_project_env

# Online profile only: load the working directory's .env. The paper profile,
# which is the default, does not read project env files.
load_project_env()
install_secret_filter()
