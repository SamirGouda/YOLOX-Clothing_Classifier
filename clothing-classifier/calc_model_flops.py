import argparse
import sys
import os
from thop import profile, clever_format
import torch
import torchvision
from util.utils import read_yaml_conf
from prettytable import PrettyTable
from pathlib import Path

import model
import dataset
import loss
from collections import OrderedDict

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf',
                        help="""path to conf dir, contains conf.yaml""",
                        default='conf/mobilenet.yaml',
                        )
    parser.add_argument('--depth',
                        help="""depth to propagete in network layer""",
                        default=2,
                        type=int,
                        )
    parser.add_argument('--log',
                        help="""path to log""",
                        default='results/mobilenet_v3_small.flops.analysis',
                        )
    try:
        params = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    return params

def find_flops_per_layer(layers2info: dict, layer: str="", depth: int = 2):
    info_dict = OrderedDict()
    for layer_name, dict_ in layers2info.items():
        m_ops, n_params, layer_type = dict_[:3]
        m_ops, n_params = clever_format([m_ops, n_params], "%.3f")
        name = f'{layer}.{layer_name}' if layer != "" else layer_name
        info_dict[name] = (m_ops, n_params, layer_type)
        next_dict = find_flops_per_layer(dict_[3], name)
        info_dict.update(next_dict)
    return info_dict

def format_layers(info_dict: dict, depth: int = 2):
    new_info_dict = {}
    for layer, info in info_dict.items():
        layer_depth = len(layer.split('.'))
        if layer_depth > depth:
            continue
        new_info_dict[layer] = info
    return new_info_dict

def create_table_for_flops(info_dict: dict):
    table = PrettyTable(['Layer', 'Type', 'MACCs', 'Params'])
    for layer, info in info_dict.items():
        macs, params, type_ = info
        table.add_row([layer, type_, macs, params])
    return table.get_string()

def write_file(file: Path, table, flops, macs, params, model):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as fd:
        fd.write(f'{table}\n\n')
        fd.write('Total FLOPs={}, MACs={}, Params={}\n\n'.format(flops, macs, params))
        fd.write(model.__repr__())

def calc_flops(log: Path, conf: dict, depth: int = 2):
    dummy_input = torch.randn(1, 3, 224, 224)
    model = conf['model_params']['backbone_model']()
    macs, params, info_dict = profile(model, inputs=(dummy_input, ), ret_layer_info=True)
    macs, params = clever_format([macs, params], "%.3f")
    # Most of modern hardware architectures uses FMA instructions for operations with tensors.
    # FMA computes a*x+b as one operation. Roughly GMACs = 0.5 * GFLOPs
    flops = f'{2* float(macs[:-1])}{macs[-1]}'
    info_dict = find_flops_per_layer(info_dict)
    info_dict = format_layers(info_dict, depth)
    info_table = create_table_for_flops(info_dict)
    write_file(log, info_table, flops, macs, params, model)
    

if __name__ == "__main__":
    print(' '.join(sys.argv))   # print command line for logging
    params = parser_arguments()
    conf = read_yaml_conf(params.conf)
    calc_flops(params.log, conf, params.depth)