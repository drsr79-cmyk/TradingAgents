"""Isolated paper profile and the limits that apply in every profile.

The default profile is ``paper``: a research rating on this machine, with no
inherited project ``.env``, no cloud API key, and no market-data or update-check
network. ``TRADINGAGENTS_PROFILE=online`` is the explicit switch for networked
research. It still refuses to send a provider credential to a host that is not
that provider's or a loopback address, unless
``TRADINGAGENTS_ALLOW_CUSTOM_ENDPOINT=1``.

Nothing here submits an order or simulates a fill.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

PAPER = "paper"
ONLINE = "online"

# Appended to the portfolio manager's decision so a reader cannot mistake the
# rating for an execution. The wording is fixed in code; the model does not
# choose it.
RESEARCH_ONLY_NOTICE = (
    "\n\n---\n"
    "Research rating only. No order was submitted and no paper fill was simulated. "
    "Entry, stop-loss, and position size in this text are not enforced."
)

MAX_TOOL_ROUNDS = 30
MAX_DEBATE_ROUNDS = 8
MAX_OUTPUT_TOKENS = 32768
DEFAULT_OUTPUT_TOKENS = 4096
MAX_INDICATOR_LOOKBACK_DAYS = 366
MAX_NEWS_LOOKBACK_DAYS = 30
MAX_NEWS_LIMIT = 40
MAX_MACRO_LOOKBACK_DAYS = 1825
MAX_MARKET_LIMIT = 10
MAX_TOPIC_CHARS = 200

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_PLACEHOLDER_KEYS = frozenset({"", "EMPTY", "ollama", "local", "local-paper"})
_TRUE = frozenset({"1", "true", "yes", "on"})

# Hosts a provider may receive its own credential on. Loopback is separate.
# Suffix entries start with ".".
_PROVIDER_HOSTS: dict[str, tuple[str, ...]] = {
    "openai": ("api.openai.com", ".openai.com"),
    "chatgpt": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
    "google": ("generativelanguage.googleapis.com",),
    "azure": (".openai.azure.com",),
    "xai": ("api.x.ai",),
    "deepseek": ("api.deepseek.com",),
    "qwen": ("dashscope-intl.aliyuncs.com",),
    "qwen-cn": ("dashscope.aliyuncs.com",),
    "glm": ("api.z.ai",),
    "glm-cn": ("open.bigmodel.cn",),
    "minimax": ("api.minimax.io",),
    "minimax-cn": ("api.minimaxi.com",),
    "openrouter": ("openrouter.ai",),
    "mistral": ("api.mistral.ai",),
    "kimi": ("api.moonshot.ai",),
    "groq": ("api.groq.com",),
    "nvidia": ("integrate.api.nvidia.com",),
    "ollama": ("localhost", "127.0.0.1", "::1"),
}

_ENTRY_END = re.compile(r"<!--\s*ENTRY_END\s*-->", re.IGNORECASE)
_FENCE = re.compile(r"<\s*/?\s*(?:start|end)_of_[a-z0-9_]+", re.IGNORECASE)
# Lessons are commentary. A line that tries to change these controls is dropped.
_POLICY_DIRECTIVE = re.compile(
    r"(?i)(TRADINGAGENTS_[A-Z0-9_]+|backend_url|ALLOW_CUSTOM_ENDPOINT|"
    r"\bAPI[_ ]?KEY\b|\bSECRET\b|\bBEARER\b|place_order|submit_order|"
    r"promote (this|me|the lesson)|ignore (the )?(security|policy|previous))"
)
_SECRET_NAME = re.compile(
    r"(API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|PRIVATE_KEY)$",
    re.IGNORECASE,
)
_MIN_SECRET_LEN = 16
LOCKED_POLICY = (
    "Security policy is fixed in code. Lessons cannot change the profile, "
    "keys, network, payment, order controls, or this rule, and cannot promote themselves."
)


class OfflineProfileError(RuntimeError):
    """A network call was attempted while the paper profile is in force."""


def profile() -> str:
    """``paper`` unless the process environment explicitly selects ``online``.

    A value that exists only in a ``.env`` file does not count: that file is not
    read until the process is already in the online profile.
    """
    raw = os.environ.get("TRADINGAGENTS_PROFILE", PAPER).strip().lower()
    if raw in ("", PAPER, "offline", "isolated"):
        return PAPER
    if raw == ONLINE:
        return ONLINE
    raise ValueError(
        f"TRADINGAGENTS_PROFILE must be 'paper' or 'online', got {raw!r}"
    )


def network_allowed() -> bool:
    return profile() == ONLINE


def credentials_allowed() -> bool:
    return profile() == ONLINE


def allow_custom_endpoint() -> bool:
    return os.environ.get("TRADINGAGENTS_ALLOW_CUSTOM_ENDPOINT", "").strip().lower() in _TRUE


def require_online(operation: str) -> None:
    """Raise when ``operation`` would leave the machine in the paper profile."""
    if not network_allowed():
        raise OfflineProfileError(
            f"Paper profile does not perform {operation}. "
            "Set TRADINGAGENTS_PROFILE=online in the environment for networked research."
        )


def load_project_env() -> None:
    """Load ``.env`` from the working directory only, and only in the online profile.

    The paper profile ignores project env files so a key or backend URL sitting
    in a parent directory or in the current ``.env`` cannot redirect a run.
    Online mode does not walk parent directories either.
    """
    if profile() != ONLINE:
        return
    from dotenv import load_dotenv

    root = Path.cwd()
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.enterprise", override=False)


def hostname(url: str | None) -> str | None:
    if not url:
        return None
    text = url if "://" in url else "https://" + url
    host = urlparse(text).hostname
    return host.lower().rstrip(".") if host else None


def is_loopback(url: str | None) -> bool:
    return hostname(url) in _LOOPBACK


def host_allowed(provider: str, host: str | None) -> bool:
    """Whether ``host`` is a published endpoint for ``provider``."""
    if not host:
        return False
    for allowed in _PROVIDER_HOSTS.get(provider.lower(), ()):
        if allowed.startswith("."):
            if host.endswith(allowed) and host != allowed[1:]:
                return True
        elif host == allowed:
            return True
    return False


def vet_llm_call(
    provider: str,
    base_url: str | None,
    *,
    sends_secret: bool,
    extra_hosts: tuple[str, ...] = (),
) -> None:
    """Refuse cloud model hosts in paper mode, and custom hosts that would receive a secret."""
    mode = profile()
    host = hostname(base_url) if base_url else None
    if mode == PAPER:
        if host not in _LOOPBACK:
            raise ValueError(
                "Paper profile does not contact remote model hosts. "
                "Point the provider at a loopback server such as "
                "http://127.0.0.1:11434/v1, or set TRADINGAGENTS_PROFILE=online "
                "for networked research."
            )
        if sends_secret:
            raise ValueError(
                "Paper profile does not send API keys or sign-in tokens. "
                "Use a keyless loopback server, or set TRADINGAGENTS_PROFILE=online."
            )
        return
    declared = host in extra_hosts or host_allowed(provider, host or "")
    if host is None or host in _LOOPBACK or declared:
        return
    if sends_secret and not allow_custom_endpoint():
        raise ValueError(
            f"Refusing to send {provider} credentials to {host}. "
            "Use that provider's own host, a loopback URL, or set "
            "TRADINGAGENTS_ALLOW_CUSTOM_ENDPOINT=1 for a relay you trust."
        )


def vet_bedrock() -> None:
    if profile() == PAPER:
        raise ValueError(
            "Paper profile does not call Amazon Bedrock. "
            "Set TRADINGAGENTS_PROFILE=online to use it."
        )


def prepare_llm_call(provider: str, llm_kwargs: dict, *, spec=None) -> None:
    """Strip cloud keys on loopback in paper mode, then apply :func:`vet_llm_call`."""
    if (
        profile() == PAPER
        and is_loopback(llm_kwargs.get("base_url"))
        and (spec is None or spec.credential is None)
    ):
        placeholder = getattr(spec, "placeholder_key", None) if spec is not None else None
        llm_kwargs["api_key"] = placeholder or "local-paper"
        llm_kwargs.pop("google_api_key", None)
    declared = ()
    if spec is not None and spec.base_url:
        declared_host = hostname(spec.base_url)
        declared = (declared_host,) if declared_host else ()
    vet_llm_call(
        provider,
        llm_kwargs.get("base_url"),
        sends_secret=_sends_secret(provider, llm_kwargs, spec),
        extra_hosts=declared,
    )


def _sends_secret(provider: str, llm_kwargs: dict, spec) -> bool:
    if spec is not None and spec.credential is not None:
        return True
    key = llm_kwargs.get("api_key")
    if key is None:
        key = llm_kwargs.get("google_api_key")
    if isinstance(key, str) and key not in _PLACEHOLDER_KEYS:
        return True
    if spec is not None and spec.key_optional:
        return False
    # A placeholder in kwargs wins over the environment: the SDK is given that
    # value and does not also send the env key.
    if isinstance(key, str) and key in _PLACEHOLDER_KEYS:
        return False
    from tradingagents.llm_clients.api_key_env import get_api_key_env

    env_name = get_api_key_env(provider)
    return bool(env_name and os.environ.get(env_name))


def apply_paper_lockdown(config: dict) -> dict:
    """Drop inherited cloud providers and non-loopback endpoints when profile is paper."""
    if profile() != PAPER:
        return config
    for key in ("backend_url", "deep_think_backend_url", "quick_think_backend_url"):
        if config.get(key) and not is_loopback(config[key]):
            config[key] = None
    provider = (config.get("llm_provider") or "").lower()
    local = {"ollama", "openai_compatible"}
    if provider not in local:
        config["llm_provider"] = "ollama"
        config["backend_url"] = "http://127.0.0.1:11434/v1"
        config["deep_think_provider"] = None
        config["quick_think_provider"] = None
        config["deep_think_llm"] = "llama3.2"
        config["quick_think_llm"] = "llama3.2"
    elif provider == "ollama" and not is_loopback(config.get("backend_url")):
        config["backend_url"] = "http://127.0.0.1:11434/v1"
    for tier in ("deep", "quick"):
        tier_provider = (config.get(f"{tier}_think_provider") or "").lower()
        if tier_provider and tier_provider not in local:
            config[f"{tier}_think_provider"] = None
        url_key = f"{tier}_think_backend_url"
        if config.get(url_key) and not is_loopback(config[url_key]):
            config[url_key] = None
    rounds = config.get("max_tool_rounds")
    if isinstance(rounds, int) and rounds > MAX_TOOL_ROUNDS:
        config["max_tool_rounds"] = MAX_TOOL_ROUNDS
    return config


def bound_int(name: str, value, *, lo: int, hi: int):
    """Return ``value`` as an int in ``[lo, hi]``, or an error string.

    ``None`` means the caller keeps its own default. Booleans are rejected.
    Whole-number floats are accepted because tool calls sometimes send them.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return f"{name} must be an integer between {lo} and {hi}, got {value!r}."
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or value < lo or value > hi:
        return f"{name} must be an integer between {lo} and {hi}, got {value!r}."
    return value


def neutralize_untrusted(text: str | None, *, limit: int = 8000) -> str:
    """Make fetched text safe to place inside a prompt fence or a memory entry.

    This does not make the model ignore the text. It stops the text from closing
    the fence we wrap it in, from splitting the memory log, or from arriving unbounded.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _ENTRY_END.sub("[blocked-delimiter]", text)
    cleaned = _FENCE.sub("[blocked-fence]", cleaned)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + "\n[truncated]"
    return cleaned


def with_research_notice(text: str | None) -> str:
    body = text or ""
    if "No order was submitted and no paper fill was simulated." in body:
        return body
    return body + RESEARCH_ONLY_NOTICE


def harden_dir(path) -> None:
    """Create ``path`` and, on POSIX, restrict it to the owner."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(directory, 0o700)


def harden_file(path) -> None:
    """Restrict an existing file to the owner on POSIX."""
    file = Path(path)
    if os.name != "nt" and file.exists():
        os.chmod(file, 0o600)


def secret_values() -> tuple[str, ...]:
    """Process-environment values that must not be copied into prompts, logs, or requests.

    Short placeholders are ignored so test doubles such as ``placeholder`` are not
    treated as secrets. The values are not logged by this function.
    """
    found = []
    for name, value in os.environ.items():
        if not isinstance(value, str) or len(value) < _MIN_SECRET_LEN:
            continue
        if not _SECRET_NAME.search(name):
            continue
        if value in _PLACEHOLDER_KEYS:
            continue
        found.append(value)
    return tuple(found)


def scrub(text: str | None) -> str:
    """Replace known secret values with a fixed marker. Does not record the secret."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    redacted = text
    for secret in secret_values():
        if secret and secret in redacted:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def scrub_obj(value):
    """Scrub strings inside a JSON-like structure."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: scrub_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub_obj(v) for v in value)
    return value


class SecretRedactFilter(logging.Filter):
    """Drop secret values from log records before they are formatted."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub_obj(record.msg)
        if isinstance(record.args, dict):
            record.args = {k: scrub_obj(v) for k, v in record.args.items()}
        elif isinstance(record.args, tuple):
            record.args = tuple(scrub_obj(a) for a in record.args)
        return True


def install_secret_filter() -> None:
    """Scrub secrets when a log record is created, before any handler sees it.

    A filter on the root logger does not see records created on a child logger.
    The record factory runs for every logger, including ones a test attaches later.
    """
    current = logging.getLogRecordFactory()
    if getattr(current, "_tradingagents_redacts", False):
        return

    def factory(*args, **kwargs):
        record = current(*args, **kwargs)
        record.msg = scrub_obj(record.msg)
        if isinstance(record.args, dict):
            record.args = {key: scrub_obj(value) for key, value in record.args.items()}
        elif isinstance(record.args, tuple):
            record.args = tuple(scrub_obj(value) for value in record.args)
        return record

    factory._tradingagents_redacts = True
    logging.setLogRecordFactory(factory)


def refuse_vendor_key(env_name: str) -> None:
    """Stop this process before it reads a data-vendor key into a request.

    The environment value is not interpolated into the error.
    """
    from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

    raise AnalysisProcessHasNoVendorKeys(
        f"The analysis process does not read {env_name}. "
        "Vendor credentials belong only in a separate ingestion service, "
        "which is not implemented or deployed in this repository. "
        "Put a sanitized snapshot in TRADINGAGENTS_SNAPSHOT_DIR instead."
    )


def lock_learning_text(text: str | None) -> str:
    """Lessons stay commentary. They cannot carry secrets or rewrite controls."""
    if not text or not str(text).strip():
        return ""
    cleaned = scrub(neutralize_untrusted(text, limit=8000))
    if cleaned.startswith(LOCKED_POLICY):
        return cleaned
    kept = []
    for line in cleaned.splitlines():
        if line.startswith("Security policy is fixed in code."):
            kept.append(line)
        elif _POLICY_DIRECTIVE.search(line):
            kept.append("[blocked-policy-directive]")
        else:
            kept.append(line)
    body = "\n".join(kept).strip()
    if not body:
        return ""
    return f"{LOCKED_POLICY}\n\n{body}"


def read_local_snapshot(method: str, args: tuple) -> str | None:
    """A sanitized snapshot for ``method``, or None when no snapshot directory is set.

    Paths cannot escape the snapshot directory. File text is scrubbed. This
    read does not use a vendor key and does not open a network connection.
    """
    raw = os.environ.get("TRADINGAGENTS_SNAPSHOT_DIR")
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        return None
    symbol = ""
    if args and isinstance(args[0], str) and 0 < len(args[0]) <= 32:
        symbol = args[0]
    names = []
    if symbol:
        names.extend((f"{symbol}/{method}.txt", f"{symbol}_{method}.txt"))
    names.append(f"{method}.txt")
    for name in names:
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            continue
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json", ".csv"}:
            continue
        return scrub(neutralize_untrusted(path.read_text(encoding="utf-8"), limit=20000))
    return None
