import os
import grpc
import requests
from eth_account import Account

from src.farcaster.builders import make_cast_add, make_signer_add 
from src.farcaster.generated.message_pb2 import CastAddBody, MessageData
from src.farcaster.client import get_insecure_client
from src.farcaster.signers import EIP712Signer, Ed25519Signer
from src.farcaster.time import get_farcaster_time

CAST_BATCHES = 1
# FARCASTER CONFIGS
FARCASTER_ID=os.getenv("FARCASTER_ID")
FARCASTER_MNEMONIC = os.getenv("FARCASTER_MNEMONIC")
FARCASTER_NETWORK = 3
HUB_ADDRESS="localhost:2283"
# WARPCAST CONFIGS
API_SECRET = os.getenv("MERKLE_SECRET")
HTTP_HEADERS = {
    "Authorization": f"Bearer {API_SECRET}"
}

if not API_SECRET:
    raise Exception("MERKLE_SECRET not set")

all_recent_casts = []
cursor = None
for i in range(CAST_BATCHES):
  url = "https://api.warpcast.com/v2/recent-casts?limit=1000"
  if cursor:
    url += f"&cursor={cursor}"
  res = requests.get(url=url, headers=HTTP_HEADERS)
  data = res.json()

  cursor = data["next"]["cursor"]
  all_recent_casts.extend(data["result"]["casts"])

cast_add_bodies = [CastAddBody(text=cast["text"]) for cast in all_recent_casts]

Account.enable_unaudited_hdwallet_features()
eth_account = Account.from_mnemonic(FARCASTER_MNEMONIC)
hub_client = get_insecure_client(HUB_ADDRESS)
signer=EIP712Signer(eth_account)
signer_add = Ed25519Signer.generate()

message_data = MessageData(
   fid=FARCASTER_ID,
   network=FARCASTER_NETWORK,
   timestamp=get_farcaster_time()
)

hub_client.SubmitMessage(
    make_signer_add(message_data, signer, signer_add)
)

for cast_add in cast_add_bodies:
    try:
        res = hub_client.SubmitMessage(
            make_cast_add(message_data, signer_add, cast_add)
        )
        print(res)
    except grpc.RpcError as e:
        print(e)
