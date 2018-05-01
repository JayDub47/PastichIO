import sys
import onnx
from onnx_coreml import convert

onnx_model = onnx.load(sys.argv[1])
preprocessing_args = {"image_scale": 1/255.0}
deprocessing_args = {"image_scale": 255.0}
coreml_model = convert(onnx_model, image_input_names=["1"], preprocessing_args=preprocessing_args, image_output_names=["211"], deprocessing_args=deprocessing_args)
coreml_model_name = sys.argv[2] + ".mlmodel"
coreml_model.save(coreml_model_name)