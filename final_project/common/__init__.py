from .env import make_env

__all__ = ['make_env', 'print_dict']


def print_dict(*dicts):
    string = []
    for d in dicts:
        for k, v in d.items():
            if abs(v) > 10:
                string.append('{}: {: 1.0f}'.format(k, v))
            elif abs(v) > 1:
                string.append('{}: {: 1.2f}'.format(k, v))
            else:
                string.append('{}: {: 1.3f}'.format(k, v))
    string = '  |  '.join(string)
    print(string)
