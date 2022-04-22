import os
import pdb
import shutil
import tempfile
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

def restore(path, modules, num_checkpoints=1, map_location=None, strict=True):
    '''
    Restore from a checkpoint

    Args:
        path - path to restore from
        modules - a dict of name to object that supports the method load_state_dict
    '''
    if not os.path.isfile(path):
        print(f'Cannot find checkpoint: {path}')
        return 0, 0

    print(f'Loading checkpoint {path}')
    state = torch.load(path, map_location=map_location)

    if 'model' in modules:
        model_state = state['model']
        root, ext = os.path.splitext(path)

        # strip any trailing digits
        base = root.rstrip(''.join(str(i) for i in range(10)))

        # determine the integer representation of the trailing digits
        idx = root[len(base):]
        start_idx = int(idx) if idx else 0

        count = 1
        for idx in range(1, num_checkpoints):
            # use the digits as the start index for loading subsequent checkpoints for averaging
            path = f'{base}{start_idx + idx}{ext}'
            if not os.path.isfile(path):
                print(f'Cannot find checkpoint: {path} Skipping it!')
                continue

            print(f'Averaging with checkpoint {path}')
            previous_state = torch.load(path, map_location=map_location)
            previous_model_state = previous_state['model']
            for name, param in model_state.items():
                param.mul_(count).add_(previous_model_state[name]).div_(count + 1)

            count += 1

        new_model_state = state['model'].copy()
        #for key in state['model']:
        #     new_key = key#.replace('module.', '')
        #   new_model_state[new_key] = state['model'][key]
            #del new_model_state[key]
        state['model'] = new_model_state

    for name, obj in modules.items():
        if isinstance(obj, nn.Module):
            obj.load_state_dict(state[name], strict=strict)
        else:
            obj.load_state_dict(state[name])
    return state['epoch'], state['step']

def checkpoint(epoch, step, modules, directory, filename='checkpoint.pt', max_checkpoints=5):
    '''
    Save a checkpoint
    Args:
        epoch - current epoch
        step - current step
        modules - a dict of name to object that supports the method state_dict
        directory - the directory to save the checkpoint file
        filename - the filename of the checkpoint
        max_checkpoints - how many checkpoints to keep
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)

    state = {
        'step': step,
        'epoch': epoch,
    }

    for name, obj in modules.items():
        state[name] = obj.state_dict()

    with tempfile.NamedTemporaryFile() as temp_checkpoint_file:
        torch.save(state, temp_checkpoint_file)

        checkpoint_path = os.path.join(directory, filename)
        if os.path.exists(checkpoint_path):
            root, ext = os.path.splitext(filename)
            for i in range(max_checkpoints - 2, -1, -1):
                previous_path = os.path.join(directory, f'{root}{i}{ext}') if i else checkpoint_path
                if os.path.exists(previous_path):
                    backup_path = os.path.join(directory, f'{root}{i+1}{ext}')
                    if os.path.exists(backup_path):
                        os.replace(previous_path, backup_path)
                    else:
                        os.rename(previous_path, backup_path)

        shutil.copy(temp_checkpoint_file.name, f'{checkpoint_path}.incomplete')
        os.rename(f'{checkpoint_path}.incomplete', checkpoint_path)

    return checkpoint_path

