import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from os.path import join
import onnx.numpy_helper as nph
import matplotlib.pyplot as plt

# build
import torchvision
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from qonnx.core.datatype import DataType
from brevitas.export import export_qonnx
from brevitas.nn import QuantConv2d, QuantReLU, QuantIdentity
from qonnx.transformation.base import Transformation

from qonnx.core.modelwrapper import ModelWrapper

from collections import OrderedDict

#verilator tests
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
import finn.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

class Sub_Weights(Transformation):
    """Convert any MultiThreshold into a standalone thresholding HLS layer."""

    def __init__(self,weights_path):
        super().__init__()
        self.weights_path = weights_path
    
    def apply(self, model):
        w_torch = torch.load(self.weights_path)
        sf, w_q, exponents = powers_of_two_quantizer(w_torch, 4)
        new_weights = exponents.to(torch.float32).numpy()
        mul_count = 0
        print(new_weights.max())
        for n in model.graph.node:
            if n.op_type == "Conv":
                model.set_initializer(n.input[1], new_weights)
                model.set_tensor_datatype(n.input[1], DataType["INT4"])
                # print('changed', n.input[1], 'int7')
            if n.op_type == "Mul":
                if mul_count == 1:
                    model.set_initializer(n.input[1], sf.numpy())
                else:
                    mul_count += 1
        return (model,False)
    

# def substitute_weights(model, weights_path):

#     # w_torch = torch.load(weights_path, map_location='cpu')['model']['module.model.0.conv.weight']
#     w_torch = torch.load(weights_path)
#     sf, w_q, exponents = powers_of_two_quantizer(w_torch, 4)
#     new_weights = exponents.numpy()
#     for n in model.graph.node:
#         if n.op_type == "Conv":
#             model.set_initializer(n.input[1], new_weights.to(torch.float32))
#             model.set_tensor_datatype(n.input[1], DataType["UINT4"])
#             # print('changed', n.input[1], 'int7')
#         if n.op_type == "Mul":
#             model.set_initializer(n.input[1], sf.numpy())
    
#     return model

def powers_of_two_quantizer(weights, bitwidth=4):
    w_module = torch.absolute(weights)
    scaling_factor = torch.max(w_module)
    sign_mask = torch.where(weights > 0, torch.ones_like(weights), torch.ones_like(weights) * (-1))

    w_scaled = w_module / scaling_factor
    min_quantized_weight_value = 2 ** ((1 - 2 ** (bitwidth - 1)) + 1)
    w_pruned = torch.where(w_scaled < min_quantized_weight_value, torch.zeros_like(weights), w_scaled)

    zero_mask = torch.where(w_pruned == 0, torch.zeros_like(weights), torch.ones_like(weights))

    w_log = torch.log2(w_pruned)

    w_rounded = torch.ceil(w_log)
    w_quant = torch.clip(w_rounded, (1 - 2 ** (bitwidth - 1)) + 1, 1)  # correction to include 0 weight (and allow for safe pruning)
    mask = (sign_mask < 0).numpy().astype(np.int32)
    exponents = np.abs(w_quant) + mask * 2**(bitwidth - 1)
    zero_exponents = torch.where(w_rounded == float('-Inf'), (2 ** (bitwidth - 1) - 1) * torch.ones_like(exponents), exponents)
    quantized = (((2 ** w_quant)) * sign_mask) * zero_mask

    return scaling_factor, quantized, zero_exponents


IN_CH = 1
OUT_CH = 3
K_SIZE = 1
BITW = 32
OUT_DATA_DIR = "testbench_python/data"
OUT_MDL_DIR = "testbench_python/model"
LYR_CFG_DIR = "testbench_python/layer_cfg"


from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.inject.enum import ScalingImplType, RestrictValueType
class Int4WeightPerTensorFloat(Int8WeightPerTensorFloat):
    bit_width = 4
    # scaling_impl_type = ScalingImplType.CONST
    # scaling_init = 0.8739659190177917
    # restrict_scaling_type = RestrictValueType.FP

def test():
    #NCHW
    # test_input = torch.randint(0, 256, (1, IN_CH, 5, 5)).float()
    # np.save(join(OUT_DATA_DIR,'test_input.npy'), test_input.numpy().transpose(0, 2, 3, 1)) # save as NHWC
    test_input = np.load(join(OUT_DATA_DIR,'test_input.npy')).transpose(0,3,1,2)#-> convert back to NCHW
    test_input = torch.from_numpy(test_input).float()

    # w_float = torch.rand(OUT_CH, IN_CH, K_SIZE, K_SIZE)
    # torch.save(w_float, join(OUT_DATA_DIR,"test_weights.pt"))
    w_float = torch.load(join(OUT_DATA_DIR,"test_weights.pt")).float()
    print(w_float)
    print("--"*10)


    sf, w_q, exponents = powers_of_two_quantizer(w_float, BITW)
    print(f"sf == {sf}")
    print(f"w_q == {w_q}")
    print(f"exponents == {exponents}")

    # test output
    conv = torch.nn.Conv2d(IN_CH, OUT_CH, K_SIZE, bias=False, stride=1, padding=1)
    # print(conv.weight)
    conv.weight = torch.nn.Parameter(w_q)
    output_quant = conv(test_input)
    print("--"*10)
    print(test_input)
    # output_quant = output_quant * sf
    print("--"*10)
    print(output_quant)
    np.save(join(OUT_DATA_DIR,'test_output.npy'), output_quant.detach().numpy())

    net = QuantConv2d(IN_CH, OUT_CH, K_SIZE, bias=False, stride=1, padding=1, weight_quant=Int4WeightPerTensorFloat)
    # print(f"Quantized weight QuantTensor:\n {net.quant_weight()} \n")  
    net.weight = torch.nn.Parameter(w_q)
    # net.weight_quant.scale().data.fill_(0.8739659190177917)
    # print(f"Original float weight tensor:\n {net.weight} \n")
    # print(f"Quantized weight QuantTensor:\n {net.quant_weight()} \n")  
    # print("--"*10)
    print(net(test_input))
    model = torch.nn.Sequential(
        OrderedDict(
            [   
                ("conv1", net)

            ]
        )
    )
    
    export_qonnx(model, test_input, join(OUT_MDL_DIR,'conv.onnx'))   

def test_pot_quantizers():
    weights_file = join(OUT_DATA_DIR,"test_weights.pt")
    weights = torch.flatten(torch.load(weights_file))
    weights[0] = 0.05
    weights = torch.linspace(0,1,10000)
    sf,quantized, exponents = powers_of_two_quantizer(weights, bitwidth=8)
    print(np.unique(exponents.numpy()))
    print(-3 << 1)
    # print(sf)
    # print(weights)
    # print(quantized)
    # print(exponents)
    plt.plot(weights,exponents)
    plt.savefig(join(OUT_DATA_DIR,"plot.png"))


import qonnx.custom_op.registry as registry
from finn.util.basic import get_rtlsim_trace_depth

def step_replace_weights(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    weights_file = join(OUT_DATA_DIR,"test_weights.pt")
    # model = substitute_weights(model,weights_file)
    model = model.transform(Sub_Weights(weights_file))
    return model


def step_change_input_type(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    return model

def step_unique_names(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def build_prehack():
    BUILD_DIR = os.environ["FINN_BUILD_DIR"]
    OUTPUT_DIR = join(BUILD_DIR,"pot_test")
    model_file = join(OUT_MDL_DIR,"conv.onnx")
    layer_file = join(LYR_CFG_DIR,"layers.json")
    BOARD = "KV260_SOM"

    build_steps_prehack = [
        "step_qonnx_to_finn",
        step_change_input_type,
        step_replace_weights,
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
        "step_specialize_layers",
        "step_transpose_decomposition",
        step_unique_names,
        "step_create_dataflow_partition",
        "step_minimize_bit_width",
        "step_hw_codegen",
        "step_hw_ipgen",
    ]

    cfg = build.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        verbose=True,
        standalone_thresholds=True,
        auto_fifo_depths=False,
        split_large_fifos=True,
        synth_clk_period_ns=10,
        specialize_layers_config_file=layer_file,
        board=BOARD,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        steps=build_steps_prehack,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)

def build_posthack():
    BUILD_DIR = os.environ["FINN_BUILD_DIR"]
    OUTPUT_DIR = join(BUILD_DIR,"pot_test")
    model_file = "/mnt/sda1/mgr/finn_build_dir/pot_test/intermediate_models/step_hw_ipgen.onnx"
    layer_file = join(LYR_CFG_DIR,"layers.json")
    BOARD = "KV260_SOM"

    build_steps_posthack = [
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile"
        # "step_make_pynq_driver",
        # "step_deployment_package",
    ]

    cfg = build.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        verbose=True,
        standalone_thresholds=True,
        auto_fifo_depths=False,
        split_large_fifos=True,
        synth_clk_period_ns=10,
        specialize_layers_config_file=layer_file,
        board=BOARD,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        steps=build_steps_posthack,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)


def test_verilator():
    BUILD_DIR = os.environ["FINN_BUILD_DIR"]
    dirc = join(BUILD_DIR,"pot_test","intermediate_models")
    os.environ["RTLSIM_TRACE_DEPTH"] = "3"
    
    #Load test input
    test_input = np.load(join(OUT_DATA_DIR,'test_input.npy')).transpose(0,3,1,2)
    #Load test output
    test_output = np.load(join(OUT_DATA_DIR,'test_output.npy'))

    print(f"test_input.shape: {test_input.shape}")

    #Create dicitonary for onnx execution
    input_dict = {"global_in": test_input.astype(np.float32)}

    #Load child model and prepare for rtlsim
    child_model = ModelWrapper(join(dirc,"step_hw_ipgen.onnx"))
    child_model = child_model.transform(SetExecMode("rtlsim"))
    child_model = child_model.transform(PrepareRTLSim())
    for node in child_model.graph.node:
        inst = registry.getCustomOp(node)
        if node.op_type == "MVAU_rtl":
            inst.set_nodeattr("rtlsim_trace","/home/bankab/"+node.op_type+".vcd")
    child_model.save(join(dirc,"step_hw_ipgen_dataflow_child.onnx"))

    # parent model
    model_for_rtlsim = ModelWrapper(join(dirc,"dataflow_parent.onnx"))
    # reference child model
    sdp_node = getCustomOp(model_for_rtlsim.graph.node[1])
    sdp_node.set_nodeattr("model", join(dirc,"step_hw_ipgen_dataflow_child.onnx"))
    model_for_rtlsim = model_for_rtlsim.transform(SetExecMode("rtlsim"))
    
    # Execute the model using the onnx executor, which will run the RTL simulation for the child model and return the outputs
    output_dict = oxe.execute_onnx(model_for_rtlsim, input_dict, True)
    print("Output from onnx executor (after RTL simulation):")
    quantization_factor = 0.1428571492433548
    print(output_dict["global_out"]/quantization_factor)
    print("Expected output (test_output):")
    print(test_output)

    print(f"Keys in output dictionary: {list(output_dict.keys())}")
    print(f"Output shape: {output_dict['global_out'].shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Error between simulated output and expected output: {np.linalg.norm(output_dict['global_out']/quantization_factor - test_output)}")


if __name__ == "__main__":
    ## FIRST, run the test to generate the ONNX model and test data
    ## dont forget to set BEHAVIORAL, in the generated MVAU_rtl wrapper
    build_prehack()
    ## THEN, run the build_posthack to generate the RTL simulation model and run the simulation
    build_posthack()
    test_verilator()

    ## Misc Tests
    # test_pot_quantizers()
    # test()
