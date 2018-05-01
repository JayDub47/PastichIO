import onnx
import sys
import torch
import torch.onnx

from Image_Transform_Net import Image_Transform_Net
from onnx_coreml import convert

pytorch_model = Image_Transform_Net().cuda()
pytorch_model.load_state_dict(torch.load(sys.argv[1]))
pytorch_model.eval()

dummy_input = torch.randn((1, 3, 512, 512)).cuda()
onnx_model_name = sys.argv[2] + ".proto"
torch.onnx.export(pytorch_model, dummy_input, onnx_model_name)

onnx_model = onnx.load(onnx_model_name)
preprocessing_args = {"image_scale":1/255.0}
deprocessing_args = {"image_scale": 255.0}
coreml_model = convert(onnx_model, image_input_names=["1"], preprocessing_args=preprocessing_args, deprocessing_args=deprocessing_args, image_output_names=["211"])
coreml_model_name = sys.argv[3] + ".mlmodel"
coreml_model.save(coreml_model_name)