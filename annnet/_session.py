"""Session, logging, and configuration for annnet, through pkg_infra.

The ecosystem shares one session handler, so a workflow that spans several
packages writes one log and reads one configuration. This module is the single
place annnet touches it.

Three rules keep a library well behaved:

1. **Importing annnet starts nothing.** A session begins when a caller asks for
   one, or when the application that embeds annnet starts its own. Importing a
   library must not create files or read configuration.
2. **A module logger never raises.** It is quiet until an application configures
   logging, and from then on it writes to the handlers of that session.
3. **A missing configuration is not an error.** Every setting has a default, and
   :func:`conf` returns it.
"""

from __future__ import annotations

from typing import Any
import logging
from pathlib import Path

PACKAGE = 'annnet'

_session: object | None = None
_session_started = False
_settings: dict | None = None


def logger(name: str) -> logging.Logger:
    """Return the logger for one module of the package.

    Call this at module level and write to the result. Use ``__name__`` as the
    name, so a reader of a log line can see which module wrote it.
    """
    try:
        from pkg_infra import module_logger
    except ImportError:
        # pkg_infra before 0.1.2 has no module_logger. The standard logger is
        # what module_logger returns anyway, so the behaviour is the same.
        return logging.getLogger(name)
    return module_logger(name)


def session(workspace: str | Path | None = None) -> object | None:
    """Start the shared session, or return the one that is already running.

    Returns ``None`` when pkg_infra cannot start a session. The package keeps
    working in that case, because a session carries logging and configuration
    and neither is needed to hold a graph.
    """
    global _session, _session_started

    if _session_started:
        return _session

    _session_started = True
    try:
        from pkg_infra import get_session

        _session = get_session(workspace=Path(workspace or Path.cwd()).resolve())
    except Exception:  # noqa: BLE001 - a package must load without a session
        _session = None
        logging.getLogger(__name__).debug(
            'No pkg_infra session for %s. Logging and configuration fall back to '
            'the standard library defaults.',
            PACKAGE,
            exc_info=True,
        )
    return _session


def settings() -> dict:
    """Return the settings of the package, read once per session.

    Reading them once matters: a configuration that names no settings for this
    package is a normal state, and asking again on every lookup would report it
    again on every lookup.
    """
    global _settings

    if _settings is not None:
        return _settings

    current = session()
    _settings = {}
    if current is not None:
        try:
            found = current.get_conf(PACKAGE)
        except Exception:  # noqa: BLE001 - a missing configuration is not an error
            found = None
        if isinstance(found, dict):
            _settings = found
    return _settings


def conf(name: str, default: Any = None) -> Any:
    """Return one setting of the package, or its default.

    The settings come from the ``annnet`` entry of the ecosystem configuration.
    """
    return settings().get(name, default)


def reset() -> None:
    """Forget the session. The next call to :func:`session` starts a new one."""
    global _session, _session_started, _settings
    _session = None
    _session_started = False
    _settings = None
