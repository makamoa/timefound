import os
import socket
import yaml

# Define the directories to be created
directories = [
    '.metadata',
    '.config'
]

# Get the home directory and project directory from environment variables
home_directory = os.path.expanduser('~')
root_directory_env = os.getenv('ROOT', 'None')
project_directory_env = os.getenv('PROJECT', 'None')
data_directory_env = os.getenv('DATA', 'None')

# Define the Hugging Face cache directory
huggingface_cache_dir = os.path.join(root_directory_env, 'cache', 'huggingface')

# Define the default settings for the YAML file
default_settings = {
    'hostname': socket.gethostname(),
    'project': os.path.abspath('.'),
    'home': home_directory,
    'root': root_directory_env,
    'projects': project_directory_env,
    'data': data_directory_env,
    'huggingface': huggingface_cache_dir,
    }

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def create_yaml_file(file_path, content):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            yaml.dump(content, file, default_flow_style=False)
        print(f"Created YAML file: {file_path}")
    else:
        print(f"YAML file already exists: {file_path}")

def create_symlink(target, link_name):
    if not os.path.exists(link_name):
        os.symlink(target, link_name)
        print(f"Created symbolic link: {link_name} -> {target}")
    else:
        print(f"Symbolic link already exists: {link_name}")

if __name__ == "__main__":
    create_directories(directories)
    create_yaml_file('.config/settings.yaml', default_settings)
    
    # Create the Hugging Face cache directory if it doesn't exist
    if not os.path.exists(huggingface_cache_dir):
        raise ValueError(f"Hugging Face cache directory not found: {huggingface_cache_dir}")
    
    # Create a symbolic link in ~/.cache to the Hugging Face cache directory
    cache_symlink = os.path.join(home_directory, '.cache', 'huggingface')
    create_symlink(huggingface_cache_dir, cache_symlink)
    
    print("Project initialization complete.")
