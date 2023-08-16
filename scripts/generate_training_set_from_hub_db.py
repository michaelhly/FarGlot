import csv
import rocksdb

from src.farcaster.generated.message_pb2 import Message

HUBBLE_ROCKS_DB_PATH=""
HUBBLE_ROCKS_DB_NAME="rocks.hub._default"

db = rocksdb.DB(f"{HUBBLE_ROCKS_DB_PATH}/{HUBBLE_ROCKS_DB_NAME}", rocksdb.Options(create_if_missing=False))
it = db.itervalues()
it.seek_to_first()

with open("data/training-set.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["hash","text","labels"])
    for serialized in it:
        try:
            m = Message()
            m.ParseFromString(serialized)
            if m.data.cast_add_body.text:
                hash = f"0x{m.hash.hex()}"
                text = m.data.cast_add_body.text
                writer.writerow([hash, text, "NEUTRAL"])
        except:
            print("skpping non message type")