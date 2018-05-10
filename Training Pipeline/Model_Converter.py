import sys
import onnx
from onnx_coreml import convert

#Load the saved, pretrained model
onnx_model = onnx.load(sys.argv[1])

'''Preprocessing and deprocessing areguments are used by the CoreML Framework
   To automatically process the data before and after it is passed through the network'''
preprocessing_args = {"image_scale": 1/255.0}
deprocessing_args = {"image_scale": 255.0}
coreml_model = convert(onnx_model, image_input_names=["1"], preprocessing_args=preprocessing_args, image_output_names=["211"], deprocessing_args=deprocessing_args)
coreml_model_name = sys.argv[2] + ".mlmodel"
coreml_model.save(coreml_model_name)