from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import NeuralCooked_pb2_grpc, NeuralCooked_pb2

class NeuralCookedServer(NeuralCooked_pb2_grpc.NeuralCookedServicer):
    script_instance = None
    async def add_seq_data(self, request, context):
        result = self.script_instance.add_seq_data(**MessageToDict(request))
        return empty_pb2.Empty()

    async def decode(self, request, context):
        result = self.script_instance.decode()
        return NeuralCooked_pb2.decodeResponse(message=result)

    async def training(self, request, context):
        result = self.script_instance.training()
        return NeuralCooked_pb2.trainingResponse(message=result)
