"""
Configuration loader for the Fact-Check Agent.

Loads API credentials from Streamlit secrets (when running via
`streamlit run` or deployed on Streamlit Community Cloud), falling back to
OS environment variables (useful for local scripts, tests, or CI that
don't go through Streamlit's secrets file).

Required keys:
    MISTRAL_API_KEY
    TAVILY_API_KEY
"""

import os
from dataclasses import dataclass
from typing import Optional

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


@dataclass(frozen=True)
class Settings:
    mistral_api_key: Optional[str]
    tavily_api_key: Optional[str]

    @property
    def is_complete(self) -> bool:
        return bool(self.mistral_api_key) and bool(self.tavily_api_key)


def _get_value(key: str) -> Optional[str]:
    """Look up a config value: Streamlit secrets first, then env vars."""
    if _HAS_STREAMLIT:
        try:
            if key in st.secrets:
                value = st.secrets[key]
                if value:
                    return str(value)
        except Exception:
            # st.secrets raises if no secrets.toml exists at all (e.g. a
            # local run before the file is created) -- fall through to env.
            pass
    return os.environ.get(key)


def load_settings() -> Settings:
    """Load settings without raising. Callers decide how to handle gaps."""
    return Settings(
        mistral_api_key=_get_value("MISTRAL_API_KEY"),
        tavily_api_key=_get_value("TAVILY_API_KEY"),
    )


def require_settings() -> Settings:
    """Load settings and raise ConfigError if anything required is missing."""
    settings = load_settings()
    missing = []
    if not settings.mistral_api_key:
        missing.append("MISTRAL_API_KEY")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if missing:
        raise ConfigError(
            "Missing required configuration: " + ", ".join(missing) + ". "
            "Set these in .streamlit/secrets.toml (local) or in the "
            "Streamlit Cloud app's Secrets settings (deployed). "
            "See .streamlit/secrets.toml.example for the expected format."
        )
    return settings
