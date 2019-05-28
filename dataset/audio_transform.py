import os
import argparse
import json
import time
import numpy as np
from torchvision import transforms
import datasets as module_dataset
import transformers as module_transformer


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
    return getattr(module, config[name]['type'])()


def save_json(x, fname, if_sort_key=False, n_indent=None):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile, sort_keys=if_sort_key, indent=n_indent)


def main(config):
    """
    Audio procesing: the transformations and directories are specified by config_audioTrans.json
    ---------------
    This parse the every entry of 'transform#' in config_audioTrans.json,
    intialize the pytorch dataset object with the specified transforms,
    and save to the specified directory in config_audioTrans.json.
    """
    # parse the transformers specified in config_audioTrans.json
    list_transformers = [get_instance(module_transformer, i, config) for i in config if 'transform' in i]
    aggr_transform = transforms.Compose(list_transformers)
    config['dataset']['args']['transform'] = aggr_transform

    # get dataset and intialize with the parsed transformers
    d = get_instance(module_dataset, 'dataset', config)
    config['dataset']['args'].pop('transform', None)  # remove once dataset is intialized, in order to save json later

    # write config file to the specified directory
    processed_audio_savePath = os.path.join(config['save_dir'], config['name'])
    if not os.path.exists(processed_audio_savePath):
        os.makedirs(processed_audio_savePath)
    print("Saving the processed audios in %s" % processed_audio_savePath)
    save_json(config, os.path.join(processed_audio_savePath, 'config.json'))

    # read, process (by transform functions in object dataset), and save
    start_time = time.time()
    for k in range(len(d)):
        audio_path = d.path_to_data[k]
        print("Transforming %d-th audio ... %s" % (k, audio_path))
        idx, y, x = d[k]

        split = audio_path.split('/')[-3]
        file_savePath = os.path.join(processed_audio_savePath, split, y)
        if not os.path.exists(file_savePath):
            os.makedirs(file_savePath)
        fname = audio_path.split('/')[-1].split('.')[0]  # replace this buggy and ugly style with Path lib
        np.save(os.path.join(file_savePath, fname), x)

    print("Processing time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Audio Transformation')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    config = json.load(open(args.parse_args().config))
    main(config)
