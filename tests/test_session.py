"""The package uses the ecosystem session, logging, and configuration.

Three properties matter for a library. Importing it must start nothing. Its
loggers must work before an application configures logging. A missing
configuration must not be an error.
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from annnet import _session


@pytest.fixture(autouse=True)
def fresh_session():
    _session.reset()
    yield
    _session.reset()


def test_importing_the_package_starts_no_session():
    """A library that reads configuration on import is a library with side effects."""
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import annnet; from annnet import _session; print(_session._session_started)',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == 'False'


def test_a_module_logger_works_before_any_session_exists():
    log = _session.logger('annnet.core.demo')
    assert isinstance(log, logging.Logger)
    assert log.name == 'annnet.core.demo'
    log.debug('this must not raise')


def test_the_same_name_returns_the_same_logger():
    assert _session.logger('annnet.core.demo') is _session.logger('annnet.core.demo')


def test_a_module_logger_reaches_the_handlers_an_application_configures(caplog):
    log = _session.logger('annnet.core.demo')
    with caplog.at_level(logging.INFO, logger='annnet.core.demo'):
        log.info('a message')
    assert 'a message' in caplog.text


def test_a_missing_setting_returns_its_default():
    assert _session.conf('no_such_setting', 'fallback') == 'fallback'


def test_the_settings_are_read_once():
    _session.settings()
    first = _session._settings
    _session.settings()
    assert _session._settings is first


def test_the_core_modules_hold_a_logger():
    from annnet.core import _attrs, _matrices, _store

    for module in (_store, _matrices, _attrs):
        assert isinstance(module.log, logging.Logger)
        assert module.log.name.startswith('annnet.core.')


def test_reset_forgets_the_session():
    _session.settings()
    assert _session._session_started
    _session.reset()
    assert not _session._session_started
    assert _session._settings is None
