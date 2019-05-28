import argparse
from torchvision import transforms
import dataset as module_dataset
import transformers as module_transformer
import json


def get_instance(module, name, config):
    """
    Get module indicated in config[name]['type'];
    If there are args to specify the module, specify in config[name]['args']
    """
    func_args = config[name]['args'] if 'args' in config[name] else None

    # if any argument specified in config[name]['args']
    if func_args:
        return getattr(module, config[name]['type'])(**func_args)
    # if not then just return the module
    return getattr(module, config[name]['type'])


def main(config):
    # parse the transformers specified in config_audioTrans.json
    list_transformers = [get_instance(module_transformer, i, config) for i in config if 'transform' in i]
    aggr_transform = transforms.Compose(list_transformers)
    config['dataset']['args']['transform'] = aggr_transform
    # get dataset and intialize with the parsed transformers
    ds = get_instance(module_dataset, 'dataset', config)
    return ds


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Audio Transformation')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    config = json.load(open(args.parse_args().config))
    d = main(config)
