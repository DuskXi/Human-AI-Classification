import subprocess


def get_gpu_info():
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].decode('utf-8').split('\n')

    out_dict = {}

    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass

    return out_dict


def get_gpu_usage():
    out_dict = get_gpu_info()
    return out_dict['Gpu'].replace('%', '').replace(' ', '')


def get_gpu_power():
    out_dict = get_gpu_info()
    return out_dict['Power Draw'].replace('W', '').replace(' ', ''), out_dict['Power Limit'].replace('W', '').replace(' ', '')


if __name__ == '__main__':
    result = get_gpu_info()
    print(result)
