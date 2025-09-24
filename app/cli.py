from typing import Annotated

import uvicorn
from typer import Option, Typer

cli = Typer()


@cli.command()
def run(
    host: Annotated[str, Option(envvar="APP_HOST")] = "localhost",
    port: Annotated[int, Option(envvar="APP_PORT")] = 8000,
    *,
    reload: Annotated[bool, Option(help="Enable hot-reload")] = False,
) -> None:
    uvicorn.run("app:app", host=host, port=port, reload=reload)


def main() -> None:
    """Shell script entrypoint."""
    cli()


if __name__ == "__main__":
    main()
