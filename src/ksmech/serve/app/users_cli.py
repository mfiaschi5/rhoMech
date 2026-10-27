"""`ksmech-serve users`: the accounts from the shell (works while the server runs: SQLite WAL).

    ksmech-serve users list
    ksmech-serve users add LOGIN --name NAME [--email E] [--role reader|user|curator|admin]
                               [--generate | --password-stdin]
    ksmech-serve users set LOGIN [--name N] [--role R] [--disable | --enable]
                               [--generate | --password-stdin]

Without `--generate` or `--password-stdin` the password is asked twice on the terminal.
`--generate` prints a random password once, as the last line. Exit 0 ok, 2 refused (a taken
login, the last admin, an unknown user, a short password).
"""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ksmech.serve.core.settings import ServeSettings

ROLES = ("reader", "user", "curator", "admin")


def add_parser(sub) -> None:
    users = sub.add_parser("users", help="list, add and edit the platform's accounts")
    action = users.add_subparsers(dest="users_command", required=True)
    action.add_parser("list", help="every account with its role")
    add = action.add_parser("add", help="a local account with a password")
    add.add_argument("login")
    add.add_argument("--name", required=True)
    add.add_argument("--email", default=None)
    add.add_argument("--role", choices=ROLES, default="user")
    edit = action.add_parser("set", help="change one account")
    edit.add_argument("login")
    edit.add_argument("--name", default=None)
    edit.add_argument("--role", choices=ROLES, default=None)
    state = edit.add_mutually_exclusive_group()
    state.add_argument("--disable", dest="disabled", action="store_const", const=True)
    state.add_argument("--enable", dest="disabled", action="store_const", const=False)
    for p in (add, edit):
        pw = p.add_mutually_exclusive_group()
        pw.add_argument("--generate", action="store_true", help="print a random password")
        pw.add_argument("--password-stdin", action="store_true", help="read it from stdin")


def _password(args: argparse.Namespace, required: bool) -> str | None:
    from ksmech.serve.accounts.passwords import generate_password

    if args.generate:
        return generate_password()
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\n")
    if not required:
        return None
    first = getpass.getpass("password: ")
    if getpass.getpass("again: ") != first:
        raise ValueError("the two passwords differ")
    return first


def _table(engine) -> None:
    from ksmech.serve.accounts import users

    print(f"{'id':>4}  {'login':<20} {'role':<8} {'state':<8} {'email':<30} providers")
    for user in users.list_users(engine):
        out = users.to_out(engine, user)
        state = "disabled" if out.disabled else "active"
        email = out.email or "-"
        print(f"{out.id:>4}  {out.login:<20} {out.role:<8} {state:<8} {email:<30} ", end="")
        print(",".join(out.providers) or "-")


def _find(engine, login: str):
    from ksmech.serve.accounts import users

    for user in users.list_users(engine):
        if user.login == login.strip().lower():
            return user
    raise ValueError(f"no user {login!r}")


def run(args: argparse.Namespace, settings: ServeSettings) -> int:
    from ksmech.serve.accounts import users
    from ksmech.serve.accounts.store import open_store
    from ksmech.serve.core.errors import ServeError

    engine = open_store(settings.accounts_db)
    try:
        if args.users_command == "list":
            _table(engine)
            return 0
        password = _password(args, required=args.users_command == "add")
        if args.users_command == "add":
            user = users.create_user(
                engine, args.login, args.name, args.role, email=args.email, password=password
            )
        else:
            patch = {"name": args.name, "role": args.role, "disabled": args.disabled}
            patch["password"] = password
            user = users.update_user(engine, int(_find(engine, args.login).id), patch)
        print(f"{user.login}: {user.role}{' (disabled)' if user.disabled else ''}")
        if args.generate:
            print(f"password: {password}")
        return 0
    except ServeError as exc:
        print(f"ksmech-serve users: [{exc.code}] {exc.message}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ksmech-serve users: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()
