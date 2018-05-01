from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, Text, Dict, Tuple
from onnx import ModelProto
from onnx.backend.base import Backend
from onnx_coreml._backend_rep import CoreMLRep
from onnx_coreml import convert
import onnx
from ._graph import _input_from_onnx_input, EdgeInfo


def _get_onnx_outputs_info(model): # type: (...) -> Dict[Text, EdgeInfo]
    """
    Takes in an onnx model and returns a dictionary 
    of onnx output names mapped to a tuple that is (output_name, type, shape)
    """
    if isinstance(model, str):
        onnx_model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        onnx_model = model

    graph = onnx_model.graph
    onnx_output_dict = {}
    for o in graph.output:
        out = _input_from_onnx_input(o)
        onnx_output_dict[out[0]] = out
    return onnx_output_dict


class CoreMLBackend(Backend):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):
        # type: (...) -> CoreMLRep
        super(CoreMLBackend, cls).prepare(model, device, **kwargs)
        coreml_model = convert(model)
        onnx_outputs_info = _get_onnx_outputs_info(model)
        return CoreMLRep(coreml_model, onnx_outputs_info, device == 'CPU')

    @classmethod
    def supports_device(cls,
                        device,  # type: Text
                        ):
        # type: (...) -> bool
        return device == 'CPU'


prepare = CoreMLBackend.prepare

run_node = CoreMLBackend.run_node

run_model = CoreMLBackend.run_model
