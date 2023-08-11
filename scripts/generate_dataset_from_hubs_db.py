import csv
import rocksdb

from src.farcaster.generated.message_pb2 import Message, MessageData, MessageType

HUBS_ROCKS_DB_PATH=""
HUBS_ROCKS_DB_NAME="rocks.hub._default"

db = rocksdb.DB(f"{HUBS_ROCKS_DB_PATH}/{HUBS_ROCKS_DB_NAME}", rocksdb.Options(create_if_missing=False))
it = db.itervalues()
it.seek_to_first()

with open("data/data-set.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["hash","text","label","score"])
    for serialized in it:
        try:
            m = Message()
            m.ParseFromString(serialized)
            if m.data.cast_add_body:
                hash = f"0x{m.hash.hex()}"
                text = m.data.cast_add_body.text
                writer.writerow([hash, text, "NEUTRAL", 0.0])
        except:
            print("skpping non message type")