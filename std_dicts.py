import json

with open('std_dicts.json', 'r') as f:
    dicts = json.load(f)
    std_dict = dicts['std_dict']
    std_bos_dict = dicts['std_bos_dict']

__all__ = ['std_dict', 'std_bos_dict']