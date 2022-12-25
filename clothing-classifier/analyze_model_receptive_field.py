import argparse
import sys
import os
import torch
import torchvision
from util.utils import read_yaml_conf
from prettytable import PrettyTable
from pathlib import Path
from torchscan import summary

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
    parser.add_argument('--log',
                        help="""path to log""",
                        default='results/mobilenet_v3_small.rf.analysis',
                        )
    try:
        params = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    return params

def find_flops_per_layer(layers2info: dict, layer: str = ""):
    info_dict = OrderedDict()
    for layer_name, dict_ in layers2info.items():
        m_ops, n_params = dict_[:2]
        # m_ops, n_params = clever_format([m_ops, n_params], "%.3f")
        name = f'{layer}.{layer_name}' if layer != "" else layer_name
        info_dict[name] = (m_ops, n_params)
        next_dict = find_flops_per_layer(dict_[2], layer_name)
        info_dict.update(next_dict)
    return info_dict

def create_table_for_flops(info_dict: dict):
    table = PrettyTable(['layer', 'MACCs', 'params'])
    for layer, info in info_dict.items():
        flops, params = info
        table.add_row([layer, flops, params])
    return table.get_string()

def write_summary_to_file(file: Path, model: torch.nn.Module):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w') as fd:
        print(summary(model, (3, 224, 224), receptive_field=True, max_depth=2), file=fd)

def summarize_model(log: Path, conf: dict):
    model = conf['model_params']['backbone_model']()
    write_summary_to_file(log, model)
    
    
if __name__ == "__main__":
    print(' '.join(sys.argv))   # print command line for logging
    params = parser_arguments()
    conf = read_yaml_conf(params.conf)
    summarize_model(params.log, conf)