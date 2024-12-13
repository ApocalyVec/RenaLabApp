from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import CVEPdummyScript_pb2_grpc, CVEPdummyScript_pb2

class CVEPdummyScriptServer(CVEPdummyScript_pb2_grpc.CVEPdummyScriptServicer):
    script_instance = None
    async def addSeqData(self, request, context):
        result = self.script_instance.addSeqData(**MessageToDict(request))
        return empty_pb2.Empty()

    async def decodeChoice(self, request, context):
        result = self.script_instance.decodeChoice()
        return CVEPdummyScript_pb2.decodeChoiceResponse(message=result)

    async def trainingModel(self, request, context):
        result = self.script_instance.trainingModel()
        return CVEPdummyScript_pb2.trainingModelResponse(message=result)
