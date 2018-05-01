import sys
import onnx

from coremltools.proto import NeuralNetwork_pb2
from onnx_coreml import convert

onnx_model = onnx.load(sys.argv[1])
preprocessing_args = {"image_scale":1/255.0}
deprocessing_args = {"image_scale": 255.0}
coreml_model = convert(onnx_model, image_input_names=["1"], preprocessing_args=preprocessing_args, deprocessing_args=deprocessing_args, image_output_names=["211"])
coreml_model.save("coreml_network.mlmodel")
