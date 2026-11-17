"""The command line of the entry point: flags, overrides, and what to run."""

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from ._build import build
from ._compose import Composer
from ._errors import ConfigError
from ._help import format_cfg, format_help
from ._interpolation import resolve
from ._multirun import Multirun, run_multirun
from ._overrides import Override, parse_override
from ._run import Entry, Job, StartDir, default_start_dir, run_job
from ._schema import RESERVED_KEY
from ._settings import Settings
from ._tree import Node

__all__ = ["run_cli"]


def _parser(program: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=program,
        add_help=False,
        allow_abbrev=False,
        usage="%(prog)s [--help] [--cfg {job,all}] [--config-name NAME] "
        "[--multirun] [key=value ...]",
    )
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--multirun", "-m", action="store_true")
    parser.add_argument("--config-name", "-cn", default=None)
    parser.add_argument("--cfg", "-c", choices=["job", "all"], default=None)
    parser.add_argument("overrides", nargs="*")
    return parser


def _settings(root: Node, now: datetime) -> Settings:
    node = root.children[RESERVED_KEY]
    resolve(root, now, only=node)
    return _built_settings(node)


def _built_settings(node: Node) -> Settings:
    settings = build(node)
    assert isinstance(settings, Settings)
    return settings


def _require_multirun(root: Node) -> None:
    """A launcher or a sweeper without `--multirun` is an error."""
    settings = root.children[RESERVED_KEY]
    for name in ("launcher", "sweeper"):
        node = settings.children.get(name)
        if node is not None and node.any_touched():
            raise ConfigError(
                f"'{RESERVED_KEY}.{name}' is set, but the {name} is only used "
                f"in a multirun. Add --multirun to the command; without it, "
                f"the job would run right here."
            )


def _start_dir(settings: Settings, now: datetime) -> StartDir:
    requested = settings.output_dir
    return StartDir(requested if requested is not None else default_start_dir(now))


def run_cli(
    entry: Entry, wrapper: Callable[..., Any], argv: Sequence[str] | None = None
) -> None:
    """Parse the command line and do what it asks for."""
    program = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "program"
    args = _parser(program).parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        code = _run(entry, wrapper, args, Path(program).stem)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        code = 1
    if code:
        sys.exit(code)


def _run(
    entry: Entry, wrapper: Callable[..., Any], args: argparse.Namespace, program: str
) -> int:
    now = datetime.now()
    overrides = [parse_override(text) for text in args.overrides]
    config_name = entry.config_name
    if args.config_name is not None:
        config_name = str(args.config_name)
    composer = Composer(entry.config_cls, entry.config_dir, Settings)

    sweeps = [override for override in overrides if override.is_sweep]
    fixed = [override for override in overrides if not override.is_sweep]

    if args.help or args.cfg is not None:
        # With --multirun, the swept values are left out: there is one config
        # per job. Without it, a sweep is the error that it always is.
        shown = fixed if args.multirun else overrides
        root = composer.compose(config_name, shown)
        if args.help:
            sys.stdout.write(format_help(composer, root))
            return 0
        sys.stdout.write(format_cfg(composer, root, own_settings=args.cfg == "all"))
        sys.stdout.flush()
        # The tree is printed with the interpolations as written. It is then
        # resolved and built as in a normal run, to report any error.
        resolve(root, now)
        build(root, skip=RESERVED_KEY)
        build(root.children[RESERVED_KEY])
        return 0

    if args.multirun:
        return _multirun(
            entry,
            wrapper,
            composer=composer,
            config_name=config_name,
            fixed=fixed,
            sweeps=sweeps,
            now=now,
            program=program,
        )

    root = composer.compose(config_name, overrides)
    _require_multirun(root)
    resolve(root, now)
    build(root, skip=RESERVED_KEY)
    settings = _built_settings(root.children[RESERVED_KEY])
    config = root.to_plain_tree()
    config.pop(RESERVED_KEY, None)
    job = Job(0, [override.text for override in overrides], config)
    run_job(entry, job, _start_dir(settings, now))
    return 0


def _multirun(
    entry: Entry,
    wrapper: Callable[..., Any],
    *,
    composer: Composer,
    config_name: str | None,
    fixed: list[Override],
    sweeps: list[Override],
    now: datetime,
    program: str,
) -> int:
    for override in sweeps:
        if override.key[0] == RESERVED_KEY:
            raise ConfigError(
                f"Error in {override.text!r}:\nnematocyte's own settings cannot be swept."
            )
    seen: dict[tuple[str, ...], Override] = {}
    for override in [*fixed, *sweeps]:
        if override.key in seen:
            raise ConfigError(
                f"The key {override.key_text!r} is given twice: "
                f"{seen[override.key].text!r} and {override.text!r}."
            )
        seen[override.key] = override

    # nematocyte's own settings are the same for every job. They are read
    # from the primary config and from the overrides under the reserved key.
    own = [override for override in fixed if override.key[0] == RESERVED_KEY]
    root = composer.compose(config_name, own)
    settings = _settings(root, now)
    run = Multirun(
        entry=entry,
        wrapper=wrapper,
        composer=composer,
        config_name=config_name,
        fixed=fixed,
        sweeps={override.key: override.value for override in sweeps},
        settings=settings,
        now=now,
        start=_start_dir(settings, now),
        program=program,
    )
    return run_multirun(run)
