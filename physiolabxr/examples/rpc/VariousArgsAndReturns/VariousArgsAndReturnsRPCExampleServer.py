from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict
import VariousArgsAndReturnsRPCExample_pb2_grpc, VariousArgsAndReturnsRPCExample_pb2

class RPCExampleServer(VariousArgsAndReturnsRPCExample_pb2_grpc.RPCExampleServicer):
    script_instance = None
    def ExampleOneArgOneReturn(self, request, context):
        result = self.script_instance.ExampleOneArgOneReturn(**MessageToDict(request))
        return VariousArgsAndReturnsRPCExample_pb2.ExampleOneArgOneReturnResponse(message=result)

    def TestRPCNoArgs(self, request, context):
        result = self.script_instance.TestRPCNoArgs()
        return VariousArgsAndReturnsRPCExample_pb2.TestRPCNoArgsResponse(message=result)

    def TestRPCNoInputNoReturn(self, request, context):
        result = self.script_instance.TestRPCNoInputNoReturn()
        return empty_pb2.Empty()

    def TestRPCNoReturn(self, request, context):
        result = self.script_instance.TestRPCNoReturn(**MessageToDict(request))
        return empty_pb2.Empty()

    def TestRPCTwoArgTwoReturn(self, request, context):
        result = self.script_instance.TestRPCTwoArgTwoReturn(**MessageToDict(request))
        return VariousArgsAndReturnsRPCExample_pb2.TestRPCTwoArgTwoReturnResponse(message0=result[0], message1=result[1])
