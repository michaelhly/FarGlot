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
DEFAULT_CLASS_VALUE = 0
DEFAULT_HUB_DB_DIR = Path.home().joinpath("apps/hubble/.rocks/rocks.hub._default")


class Classifier:
    def __init__(
        self, name: str, default_value: Optional[Union[int, str]] = DEFAULT_CLASS_VALUE
    ):
        self.name = name
        self.default_value = default_value


app = typer.Typer()


@app.command()
def init():
    FARGLOT_DIR.mkdir(exist_ok=True)
    __write_paths_to_conf(FARGLOT_DIR.joinpath("classifers.json"), DEFAULT_HUB_DB_DIR)
    typer.echo(f"FarGlot configs initialized at {FARGLOT_CONF_PATH}")


@app.command(help="Set path to your Hub's RocksDB directory")
def set_hub_db_path(db_path: str):
    hub_db_path = Path(db_path)
    if not hub_db_path.exists() or not hub_db_path.is_dir():
        raise ValueError(f"${db_path} is not a valid directory")

    classifiers_path, _ = __read_paths_from_conf()
    __write_paths_to_conf(classifiers_path, hub_db_path)
    typer.echo(f"Hub DB path updated to {db_path}")


@app.command(help="Set path to JSON file with class headers")
def set_classifiers_path(classifiers_path: str):
    lp = Path(classifiers_path)
    if not lp.exists() or lp.is_dir():
        raise ValueError(f"${classifiers_path} is not a valid file")

    _, db_path = __read_paths_from_conf()
    __write_paths_to_conf(lp, db_path)
    typer.echo(f"Classifiers path updated to {classifiers_path}")


@app.command(help="Generate a new training set from your Hub")
def new_training_set(out: Optional[str] = None):
    classifiers_path, db_path = __read_paths_from_conf()
    if not classifiers_path.exists():
        raise FileNotFoundError(f"{classifiers_path}")

    typer.echo(f"Pulling classifers from ${classifiers_path} ...")
    with classifiers_path.open() as f:
        try:
            classifers_json = json.load(f)
        except:
            raise InvalidFileException(
                f"failed to decode classifers from {classifiers_path}"
            )

        if type(classifers_json) == list:
            classifers: List[dict] = classifers_json
            classifers = [
                Classifier(
                    classifier["name"],
                    classifier.get("default_value", DEFAULT_CLASS_VALUE),
                )
                for classifier in classifers
            ]
        elif type(classifers_json) == dict:
            single_classifer: dict = classifers_json
            classifers = [
                Classifier(
                    single_classifer["name"],
                    single_classifer.get("default_value", DEFAULT_CLASS_VALUE),
                )
            ]
        else:
            raise ValueError("f{classifers_json} is not valid")

    __generate_training_set_from_rocks_db(db_path, classifers, out)


def __generate_training_set_from_rocks_db(
    hub_db_path: Path, classifiers: List[Classifier], out: Optional[str] = None
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

        class_names = ["hash", "text"]
        class_names.extend([classifier.name for classifier in classifiers])
        csv_writer.writerow(class_names)

        default_class_values = [classifier.default_value for classifier in classifiers]
        for serialized in it:
            try:
                m = Message()
                m.ParseFromString(serialized)
                if m.data.cast_add_body.text:
                    hash = f"0x{m.hash.hex()}"
                    text = m.data.cast_add_body.text
                    row = [hash, text]
                    row.extend(default_class_values)
                    csv_writer.writerow(row)
            except:
                # typer.echo("skpping non message type")
                continue

    typer.echo(f"training-set.csv available at ${out_path.absolute()}")


def __read_paths_from_conf() -> Tuple[Path, Path]:
    with FARGLOT_CONF_PATH.open(mode="r") as conf:
        classifiers_path = conf.readline().strip("\n")
        db_path = conf.readline().strip("\n")
    return Path(classifiers_path), Path(db_path)


def __write_paths_to_conf(classifiers_path: Path, db_path: Path):
    with FARGLOT_CONF_PATH.open(mode="w") as conf:
        conf.writelines([str(classifiers_path) + "\n", str(db_path)])


if __name__ == "__main__":
    app()
