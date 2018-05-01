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
coreml_model = convert(onnx_model, add_custom_layers=True)
coreml_model_name = sys.argv[3] + ".mlmodel"
coreml_model.save(coreml_model_name)