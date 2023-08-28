from typing import Optional
from typing_extensions import Self
from transformers import PretrainedConfig

from farglot.__farcaster.client import get_ssl_client
from farglot.__farcaster.generated.message_pb2 import CastId, Message
from farglot.__farcaster.generated.request_response_pb2 import (
    FidRequest,
    MessagesResponse,
)
from farglot.__farcaster.generated.rpc_pb2_grpc import (
    HubServiceStub,
)
from farglot.analyzer import (
    AnalyzerForSequenceClassification,
    AnalyzerForTokenClassification,
    BaseAnalyzer,
)


class CastAnalyzer:
    def __init__(self, hub_client: HubServiceStub, analyzer: BaseAnalyzer) -> Self:
        self.analyzer = analyzer
        self.hub_client = hub_client

    @classmethod
    def sequence_analzyer_from_model_name(
        cls,
        hub_address: str,
        model_name: str,
        config: Optional[PretrainedConfig] = None,
    ) -> Self:
        return cls(
            hub_client=get_ssl_client(hub_address),
            analyzer=AnalyzerForSequenceClassification.from_model_name(
                model_name=model_name, config=config
            ),
        )

    @classmethod
    def token_analyzer_from_model_name(
        cls,
        hub_address: str,
        model_name: str,
        config: Optional[PretrainedConfig] = None,
    ) -> Self:
        return cls(
            hub_client=get_ssl_client(hub_address),
            analyzer=AnalyzerForTokenClassification.from_model_name(
                model_name=model_name, config=config
            ),
        )

    def predict_cast(self, fid: int, hash_hex: str):
        message: Message = self.hub_client.GetCast(
            CastId(fid=fid, hash=bytes.fromhex(hash_hex))
        )
        return self.analyzer.predict([message.data.cast_add_body.text])

    def predict_casts_by_fid(self, fid: int):
        request = FidRequest(fid=fid)
        response: MessagesResponse = self.hub_client.GetCastsByFid(request)
        inputs = [
            m.data.cast_add_body.text
            for m in response.messages
            if m.data.cast_add_body.text
        ]
        return self.analyzer.predict(inputs)
