from pathlib import Path
from typing import List, Optional, Tuple
import csv
import json
import typer
import rocksdb

from farglot.__farcaster.generated.message_pb2 import Message


class Label:
    def __init__(self, name: str, default_value: Optional[str] = None):
        self.name = name
        self.default_value = default_value


class Configs:
    def __init__(self):
        self.farglot_dir = Path.home().joinpath(".farglot")
        self.default_hub_db_path = Path.home().joinpath("apps/hubble/.rocks/rocks.hub._default")


app = typer.Typer()
configs = Configs()


@app.command()
def init():
    configs.farglot_dir.mkdir(exist_ok=True)
    __write_paths_to_conf(
        configs.farglot_dir.joinpath("labels.json"),
        configs.default_hub_db_path
    )

@app.command()
def set_hub_db_path(db_path: str):
    hub_db_path = Path(db_path)
    if not hub_db_path.exists() or not hub_db_path.is_dir():
        raise ValueError(f"${db_path} is not a valid directory")

    label_path, _ = __read_paths_from_conf()
    __write_paths_to_conf(label_path, hub_db_path)

@app.command()
def set_labels_path(label_path: str):
    lp = Path(label_path)
    if not lp.exists() or lp.is_dir():
        raise ValueError(f"${label_path} is not a valid file")

    _, db_path = __read_paths_from_conf()
    __write_paths_to_conf(lp, db_path)


@app.command()
def new_training_set(out: Optional[str] = None):
    labels_path, db_path = __read_paths_from_conf()
    if not labels_path.exists():
        raise FileNotFoundError(f"{str(labels_path)}")

    with labels_path.open() as f:
        labels_json = json.load(f)
        if type(labels_json) == list:
            labels_list: List[dict] = labels_json
            labels = [
                Label(label["name"], label.get("default_value"))
                for label in labels_list
            ]
        elif type(labels_json) == dict:
            labels_single_dict: dict = labels_json
            labels = [
                Label(
                    labels_single_dict["name"], labels_single_dict.get("default_value")
                )
            ]
        else:
            raise ValueError("f{labels_json} is not valid")

    print(labels)
    __generate_training_set_from_rocks_db(db_path, labels, out)


def __generate_training_set_from_rocks_db(
    hub_db_path: Path,
    labels: List[Label],
    out: Optional[str] = None
):
    db = rocksdb.DB(
        str(hub_db_path),
        rocksdb.Options(create_if_missing=False),
    )
    it = db.itervalues()
    it.seek_to_first()

    out_path = Path(out) if out else configs.farglot_dir.joinpath("training-set.csv")
    with out_path.open(mode="w") as f:
        csv_writer = csv.writer(f)

        label_names = ["hash", "text"]
        label_names.extend([label.name for label in labels])
        csv_writer.writerow(label_names)

        default_label_values = [label.default_value for label in labels]
        for serialized in it:
            try:
                m = Message()
                m.ParseFromString(serialized)
                if m.data.cast_add_body.text:
                    hash = f"0x{m.hash.hex()}"
                    text = m.data.cast_add_body.text
                    row = [hash, text]
                    row.extend(default_label_values)
                    csv_writer.writerow(row)
            except:
                print("skpping non message type")

def __read_paths_from_conf() -> Tuple[Path, Path]:
    conf_path = configs.farglot_dir.joinpath("conf")
    with conf_path.open(mode="r") as conf:
        labels_path = conf.readline().strip("\n")
        db_path = conf.readline().strip("\n")
    return Path(labels_path), Path(db_path)


def __write_paths_to_conf(labels_path: Path, db_path: Path):
    conf_path = configs.farglot_dir.joinpath("conf")
    with conf_path.open(mode="w") as conf:
        conf.writelines([str(labels_path) + "\n", str(db_path)])

if __name__ == "__main__":
    app()
