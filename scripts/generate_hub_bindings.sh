set -e

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
FARCASTER_DIR="$../.farcaster"
GENERATED_DIR="$SCRIPT_DIR/farcaster/generated"

# https://stackoverflow.com/questions/600079/how-do-i-clone-a-subdirectory-only-of-a-git-repository/52269934#52269934
svn export --force https://github.com/farcasterxyz/hub-monorepo/trunk/protobufs/schemas $FARCASTER_DIR/schemas
python3 -m grpc_tools.protoc \
    -I$FARCASTER_DIR/schemas \
    --python_out=$GENERATED_DIR  --pyi_out=$GENERATED_DIR --grpc_python_out=$GENERATED_DIR  \
    $FARCASTER_DIR/schemas/*.proto
