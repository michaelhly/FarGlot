from pathlib import Path
from plistlib import InvalidFileException
from typing import List, Optional, Tuple, Union
import csv
import json
import typer
import rocksdb

from farglot.__farcaster.generated.message_pb2 import Message

FARGLOT_DIR = Path.home().joinpath(".farglot")
FARGLOT_CONF_PATH = FARGLOT_DIR.joinpath("conf")
DEFAULT_COLUMN_VALUE = 0
DEFAULT_HUB_DB_DIR = Path.home().joinpath("apps/hubble/.rocks/rocks.hub._default")


class Column:
    def __init__(
        self, name: str, default_value: Optional[Union[int, str]] = DEFAULT_COLUMN_VALUE
    ):
        self.name = name
        self.default_value = default_value


app = typer.Typer()


@app.command()
def init():
    FARGLOT_DIR.mkdir(exist_ok=True)
    __write_paths_to_conf(FARGLOT_DIR.joinpath("columns.json"), DEFAULT_HUB_DB_DIR)
    typer.echo(f"FarGlot configs initialized at {str(FARGLOT_CONF_PATH)}")


@app.command()
def set_hub_db_path(db_path: str):
    hub_db_path = Path(db_path)
    if not hub_db_path.exists() or not hub_db_path.is_dir():
        raise ValueError(f"${db_path} is not a valid directory")

    columns_path, _ = __read_paths_from_conf()
    __write_paths_to_conf(columns_path, hub_db_path)
    typer.echo(f"Hub DB path updated to {db_path}")


@app.command()
def set_columns_path(columns_path: str):
    lp = Path(columns_path)
    if not lp.exists() or lp.is_dir():
        raise ValueError(f"${columns_path} is not a valid file")

    _, db_path = __read_paths_from_conf()
    __write_paths_to_conf(lp, db_path)
    typer.echo(f"Columns path updated to {columns_path}")


@app.command()
def new_training_set(out: Optional[str] = None):
    columns_path, db_path = __read_paths_from_conf()
    if not columns_path.exists():
        raise FileNotFoundError(f"{str(columns_path)}")

    typer.echo(f"Pulling columns from ${str(columns_path)} ...")
    with columns_path.open() as f:
        try:
            columns_json = json.load(f)
        except:
            raise InvalidFileException(
                f"failed to decode columns from {str(columns_path)}"
            )

        if type(columns_json) == list:
            columns_list: List[dict] = columns_json
            columns = [
                Column(
                    column["name"], column.get("default_value", DEFAULT_COLUMN_VALUE)
                )
                for column in columns_list
            ]
        elif type(columns_json) == dict:
            column_single_dict: dict = columns_json
            columns = [
                Column(
                    column_single_dict["name"],
                    column_single_dict.get("default_value", DEFAULT_COLUMN_VALUE),
                )
            ]
        else:
            raise ValueError("f{columns_json} is not valid")

    __generate_training_set_from_rocks_db(db_path, columns, out)


def __generate_training_set_from_rocks_db(
    hub_db_path: Path, columns: List[Column], out: Optional[str] = None
):
    db = rocksdb.DB(
        str(hub_db_path),
        rocksdb.Options(create_if_missing=False),
    )

    typer.echo(f"Pulling casts from ${str(hub_db_path)} ...")
    it = db.itervalues()
    it.seek_to_first()

    out_path = Path(out) if out else FARGLOT_DIR.joinpath("training-set.csv")
    with out_path.open(mode="w") as f:
        csv_writer = csv.writer(f)

        column_names = ["hash", "text"]
        column_names.extend([column.name for column in columns])
        csv_writer.writerow(column_names)

        default_column_values = [column.default_value for column in columns]
        for serialized in it:
            try:
                m = Message()
                m.ParseFromString(serialized)
                if m.data.cast_add_body.text:
                    hash = f"0x{m.hash.hex()}"
                    text = m.data.cast_add_body.text
                    row = [hash, text]
                    row.extend(default_column_values)
                    csv_writer.writerow(row)
            except:
                # typer.echo("skpping non message type")
                continue

    typer.echo(f"training-set.csv available at ${out_path.absolute()}")


def __read_paths_from_conf() -> Tuple[Path, Path]:
    with FARGLOT_CONF_PATH.open(mode="r") as conf:
        columns_path = conf.readline().strip("\n")
        db_path = conf.readline().strip("\n")
    return Path(columns_path), Path(db_path)


def __write_paths_to_conf(columns_path: Path, db_path: Path):
    with FARGLOT_CONF_PATH.open(mode="w") as conf:
        conf.writelines([str(columns_path) + "\n", str(db_path)])


if __name__ == "__main__":
    app()
