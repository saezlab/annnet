"""Package metadata helpers."""

from __future__ import annotations

import sys
from html import escape
from typing import Any
from pathlib import Path
from importlib import metadata as importlib_metadata
from collections import OrderedDict
from dataclasses import dataclass

from ._optional_components import (
    IO_MODULES,
    PLOT_BACKENDS,
    GRAPH_BACKENDS,
    DATAFRAME_BACKENDS,
    component_names,
    component_status,
)

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    'get_metadata',
    'get_latest_version',
    'metadata',
    'info',
    'supports_html',
    '__title__',
    '__version__',
    '__author__',
    '__authors__',
    '__maintainers__',
    '__license__',
]

_FALLBACK_VERSION = '0.1.0'
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT_PATH = _PROJECT_ROOT / 'pyproject.toml'


def _default_graph_backend(values: dict[str, dict[str, str]]) -> str:
    for name in component_names(GRAPH_BACKENDS):
        if values.get(name, {}).get('available') == 'yes':
            return name
    return 'none'


def _default_plot_backend(values: dict[str, dict[str, str]]) -> str:
    for name in component_names(PLOT_BACKENDS):
        if values.get(name, {}).get('available') == 'yes':
            return name
    return 'none'


def _project_optional_dependencies(meta: dict[str, Any]) -> dict[str, list[str]]:
    project = meta.get('full_metadata', {}).get('project', {})
    return project.get('optional-dependencies', {}) if isinstance(project, dict) else {}


def _optional_dependency_bundles(meta: dict[str, Any]) -> OrderedDict[str, list[str]]:
    optional = _project_optional_dependencies(meta)
    aggregate = {'io', 'backends', 'plot', 'bio', 'storage', 'all', 'dev'}
    return OrderedDict((name, deps) for name, deps in optional.items() if name in aggregate)


def _author_links(authors: list[str]) -> str:
    rendered: list[str] = []
    for author in authors:
        if '<' in author and author.endswith('>'):
            name, email = author.rsplit('<', 1)
            name = name.strip()
            email = email[:-1].strip()
            rendered.append(
                f'{escape(name)} '
                f'<a href="mailto:{escape(email)}" style="text-decoration:none" title="{escape(email)}">&#9993;</a>'
            )
        else:
            rendered.append(escape(author))
    return ', '.join(rendered) if rendered else 'n/a'


class DisplayInspector:
    """Object used to probe whether the frontend renders HTML."""

    def __init__(self) -> None:
        self.status: str | None = None

    def _repr_html_(self) -> str:
        self.status = 'HTML'
        return ''

    def __repr__(self) -> str:
        self.status = 'Plain'
        return ''


def supports_html() -> bool:
    """Best-effort check for HTML-capable frontends."""

    if 'marimo' in sys.modules:
        return True

    if 'IPython' not in sys.modules or 'IPython.display' not in sys.modules:
        return False

    try:
        from IPython.display import display
    except Exception:  # pragma: no cover - import edge case  # noqa: BLE001
        return False

    inspector = DisplayInspector()
    display(inspector)
    return inspector.status == 'HTML'


def get_latest_version(
    url: str = 'https://raw.githubusercontent.com/saezlab/annnet/main/pyproject.toml',
    timeout: int = 5,
) -> str | None:
    """Fetch the latest version declared on the default branch."""

    import re
    from urllib.parse import urlparse
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'}:
        return None

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
            content = response.read().decode()
        match = re.search(r'version\s*=\s*"(.*)"', content)
        if match:
            return match.group(1)
    except Exception:  # noqa: BLE001
        return None

    return None


@dataclass(frozen=True)
class AnnNetInfo:
    metadata: dict[str, Any]
    graph_backends: dict[str, dict[str, str]]
    plot_backends: dict[str, dict[str, str]]
    tabular_backends: dict[str, dict[str, str]]
    io_modules: dict[str, dict[str, str]]

    def _section_message(self, values: dict[str, dict[str, str]]) -> str:
        parts = []
        for name, details in values.items():
            suffix = f' ({details["install"]})' if details['available'] == 'no' else ''
            parts.append(f'{name}: {details["available"]}{suffix}')
        return '; '.join(parts) if parts else 'n/a'

    def _info(self) -> OrderedDict[str, dict[str, Any]]:
        urls = self.metadata.get('urls', {})
        info: OrderedDict[str, dict[str, Any]] = OrderedDict()
        info['annnet_version'] = {
            'title': 'Installed version',
            'message': f'v{self.metadata.get("version", _FALLBACK_VERSION)}',
            'value': self.metadata.get('version', _FALLBACK_VERSION),
        }
        info['license'] = {
            'title': 'License',
            'message': self.metadata.get('license', 'BSD-3-Clause'),
            'value': self.metadata.get('license', 'BSD-3-Clause'),
        }
        if self.metadata.get('author'):
            info['authors'] = {
                'title': 'Authors',
                'message': self.metadata['author'],
                'value': self.metadata['authors'],
            }
        info['graph_backends'] = {
            'title': 'Available graph backends',
            'message': self._section_message(self.graph_backends),
            'value': self.graph_backends,
        }
        info['plot_backends'] = {
            'title': 'Available plot backends',
            'message': self._section_message(self.plot_backends),
            'value': self.plot_backends,
        }
        info['tabular_backends'] = {
            'title': 'Available tabular backends',
            'message': self._section_message(self.tabular_backends),
            'value': self.tabular_backends,
        }
        info['io_modules'] = {
            'title': 'Available I/O modules',
            'message': self._section_message(self.io_modules),
            'value': self.io_modules,
        }
        if urls.get('Repository'):
            info['repo_url'] = {
                'title': 'Repository',
                'message': urls['Repository'],
                'value': urls['Repository'],
            }
        if urls.get('Documentation'):
            info['docs_url'] = {
                'title': 'Documentation',
                'message': urls['Documentation'],
                'value': urls['Documentation'],
            }
        info['installed_path'] = {
            'title': 'Installed path',
            'message': str(_PROJECT_ROOT),
            'value': str(_PROJECT_ROOT),
        }
        return info

    def __str__(self) -> str:
        return '\n'.join(f'{item["title"]}: {item["message"]}' for item in self._info().values())

    __repr__ = __str__

    def _repr_html_(self) -> str:
        urls = self.metadata.get('urls', {})
        optional_bundles = _optional_dependency_bundles(self.metadata)

        def render_status_chips(
            status_map: OrderedDict[str, bool] | dict[str, bool], *, show_icon: bool = True
        ) -> str:
            chips = []
            for name, enabled in status_map.items():
                icon = '&#10003;' if enabled else '&#10007;'
                color = '#1a7f37' if enabled else '#cf222e'
                icon_html = (
                    f"<span style='color:{color};font-weight:700'>{icon}</span>"
                    if show_icon
                    else ''
                )
                chips.append(
                    f"<span style='display:inline-flex;align-items:center;gap:0.28rem;"
                    'padding:0.12rem 0.45rem;margin:0.08rem 0.35rem 0.08rem 0;'
                    "border:1px solid #d0d7de;border-radius:999px;'>"
                    f'{icon_html}'
                    f'<span>{escape(name)}</span>'
                    '</span>'
                )
            return ''.join(chips) if chips else "<span style='color:#57606a'>none</span>"

        def render_named_group_chips(groups: dict[str, list[str]]) -> str:
            if not groups:
                return "<span style='color:#57606a'>none</span>"
            chips = []
            for name, deps in groups.items():
                title = ', '.join(deps)
                chips.append(
                    f'<span title="{escape(title)}" style=\'display:inline-flex;align-items:center;'
                    'padding:0.12rem 0.45rem;margin:0.08rem 0.35rem 0.08rem 0;'
                    "border:1px solid #d0d7de;border-radius:999px;'>"
                    f'<span>{escape(name)}</span>'
                    '</span>'
                )
            return ''.join(chips)

        def render_value(title: str, value: str) -> str:
            if title in {'Repository', 'Documentation'}:
                url = escape(value)
                return f'<a href="{url}">{url}</a>'
            if title == 'Authors':
                return _author_links(self.metadata.get('authors', []))
            if title == 'Installable bundles':
                return render_named_group_chips(optional_bundles)
            if title == 'Default graph backend':
                return render_status_chips(
                    OrderedDict(
                        [(_default_graph_backend(self.graph_backends), True)]
                        if _default_graph_backend(self.graph_backends) != 'none'
                        else [('none', False)]
                    ),
                    show_icon=False,
                )
            if title == 'Default plot backend':
                return render_status_chips(
                    OrderedDict(
                        [(_default_plot_backend(self.plot_backends), True)]
                        if _default_plot_backend(self.plot_backends) != 'none'
                        else [('none', False)]
                    ),
                    show_icon=False,
                )
            if title in {
                'Graph backends',
                'Plot backends',
                'Tabular data backends',
                'I/O modules',
            }:
                if title == 'Graph backends':
                    return render_status_chips(
                        OrderedDict(
                            (name, details['available'] == 'yes')
                            for name, details in self.graph_backends.items()
                        )
                    )
                if title == 'Plot backends':
                    return render_status_chips(
                        OrderedDict(
                            (name, details['available'] == 'yes')
                            for name, details in self.plot_backends.items()
                        )
                    )
                if title == 'Tabular data backends':
                    return render_status_chips(
                        OrderedDict(
                            (name, details['available'] == 'yes')
                            for name, details in self.tabular_backends.items()
                        )
                    )
                return render_status_chips(
                    OrderedDict(
                        (name, details['available'] == 'yes')
                        for name, details in self.io_modules.items()
                    )
                )
            return escape(value)

        rows = [
            ('Version', f'v{self.metadata.get("version", _FALLBACK_VERSION)}'),
            ('License', self.metadata.get('license', 'BSD-3-Clause')),
        ]

        if self.metadata.get('author'):
            rows.append(('Authors', self.metadata['author']))
        if urls.get('Repository'):
            rows.append(('Repository', urls['Repository']))
        if urls.get('Documentation'):
            rows.append(('Documentation', urls['Documentation']))
        rows.append(('Installed path', str(_PROJECT_ROOT)))
        rows.extend(
            [
                ('Default graph backend', _default_graph_backend(self.graph_backends)),
                ('Default plot backend', _default_plot_backend(self.plot_backends)),
                ('Graph backends', ''),
                ('Plot backends', ''),
                ('Tabular data backends', ''),
                ('I/O modules', ''),
                ('Installable bundles', ''),
            ]
        )

        table_rows = []
        for title, value in rows:
            rendered = render_value(title, str(value))
            table_rows.append(
                '<tr>'
                f"<th style='text-align:right;vertical-align:top;padding:0.2rem 0.9rem 0.2rem 0;white-space:nowrap;width:10.5rem;'>{escape(title)}</th>"
                f"<td style='text-align:left;padding:0.2rem 0'>{rendered}</td>"
                '</tr>'
            )

        return (
            '<div style=\'font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            'border:1px solid #d0d7de;border-radius:12px;padding:1rem 1.1rem;'
            "max-width:820px;background:#fff;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:0.6rem;'>"
            '<div>'
            "<div style='font-size:1.05rem;font-weight:700;'>annnet</div>"
            f"<div style='font-size:0.92rem;color:#57606a;'>Environment summary for this annnet v{escape(str(self.metadata.get('version', _FALLBACK_VERSION)))}</div>"
            '</div>'
            '</div>'
            "<table style='border-collapse:collapse;width:100%;'>"
            '<tbody>'
            f'{"".join(table_rows)}'
            '</tbody></table>'
            '</div>'
        )

    def _mime_(self) -> tuple[str, str]:
        return 'text/html', self._repr_html_()

    def to_html(self) -> str:
        """Return the HTML representation explicitly."""

        return self._repr_html_()


def _normalize_people(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []

    people: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get('name')
            email = entry.get('email')
            if name and email:
                people.append(f'{name} <{email}>')
            elif name:
                people.append(str(name))
            elif email:
                people.append(str(email))
        elif isinstance(entry, str):
            people.append(entry)

    return people


def _normalize_license(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get('text'), str):
            return value['text']
        if isinstance(value.get('file'), str):
            return value['file']
    return None


def _metadata_from_pyproject() -> dict[str, Any]:
    if not _PYPROJECT_PATH.exists():
        return {}

    with _PYPROJECT_PATH.open('rb') as fh:
        pyproject = tomllib.load(fh)

    project = pyproject.get('project', {})
    authors = _normalize_people(project.get('authors'))
    maintainers = _normalize_people(project.get('maintainers'))

    meta = {
        'name': project.get('name'),
        'version': project.get('version'),
        'authors': authors,
        'author': ', '.join(authors) if authors else None,
        'maintainers': maintainers,
        'license': _normalize_license(project.get('license')),
        'urls': project.get('urls', {}),
        'full_metadata': pyproject,
    }

    return {k: v for k, v in meta.items() if v is not None}


def _metadata_from_installed_package() -> dict[str, Any]:
    try:
        pkg_meta = importlib_metadata.metadata('annnet')
    except importlib_metadata.PackageNotFoundError:
        return {}

    authors: list[str] = []
    if pkg_meta.get('Author'):
        authors.append(pkg_meta['Author'])
    if pkg_meta.get('Author-email') and pkg_meta['Author-email'] not in authors:
        authors.append(pkg_meta['Author-email'])

    maintainers: list[str] = []
    if pkg_meta.get('Maintainer'):
        maintainers.append(pkg_meta['Maintainer'])
    if pkg_meta.get('Maintainer-email') and pkg_meta['Maintainer-email'] not in maintainers:
        maintainers.append(pkg_meta['Maintainer-email'])

    meta = {
        'name': pkg_meta.get('Name'),
        'version': pkg_meta.get('Version'),
        'authors': authors,
        'author': ', '.join(authors) if authors else None,
        'maintainers': maintainers,
        'license': pkg_meta.get('License'),
        'summary': pkg_meta.get('Summary'),
        'full_metadata': dict(pkg_meta.items()),
    }

    return {k: v for k, v in meta.items() if v is not None}


def get_metadata() -> dict[str, Any]:
    """Return normalized package metadata."""

    meta = _metadata_from_pyproject() or _metadata_from_installed_package()
    meta.setdefault('name', 'annnet')
    meta.setdefault('version', _FALLBACK_VERSION)
    meta.setdefault('authors', [])
    meta.setdefault('maintainers', [])
    meta.setdefault('author', ', '.join(meta['authors']) if meta['authors'] else None)
    meta.setdefault('license', 'BSD-3-Clause')
    return meta


metadata = get_metadata()
__title__ = metadata['name']
__version__ = metadata['version']
__author__ = metadata.get('author')
__authors__ = metadata['authors']
__maintainers__ = metadata['maintainers']
__license__ = metadata['license']


def info() -> AnnNetInfo:
    """Return a human-readable package component summary."""

    return AnnNetInfo(
        metadata=get_metadata(),
        graph_backends=component_status(GRAPH_BACKENDS),
        plot_backends=component_status(PLOT_BACKENDS),
        tabular_backends=component_status(DATAFRAME_BACKENDS),
        io_modules=component_status(IO_MODULES),
    )
