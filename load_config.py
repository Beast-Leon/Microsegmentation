import yaml
import os

def load_yaml(file_address):
    with open(file_address, 'r') as file:
        content = yaml.load_all(file, Loader = yaml.FullLoader)
        return list(content)

def dump_yaml(input_list, file_address):
    with open(file_address, 'w') as file:
        data = yaml.dump(input_list, file, default_flow_style = False)

def safe_load_yaml(file_address):
    with open(file_address, 'r') as file:
        try:
            config_settings = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
    return config_settings 
if __name__ == "__main__":
    config = safe_load_yaml("general_config.yaml")
    print(config['model']['decision_tree'])