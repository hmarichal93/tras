import json
import sys

def load_json(filepath: str) -> dict:
    """
    Load json utility.
    :param filepath: file to json file
    :return: the loaded json as a dictionary
    """
    with open(str(filepath), 'r') as f:
        data = json.load(f)
    return data


def write_binary_file(binary_content: bytes, output_file:str) -> None:

    with open(output_file, "wb") as f:
        f.write(binary_content)
    return

def read_file_as_binary(file_path: str) -> bytes:
    """
    Read file as binary
    :param file_path: file path to read
    :return: binary content
    """
    with open(file_path, "rb") as f:
        content = f.read()
    return content

def write_json(dict_to_save: dict, filepath: str) -> None:
    """
    Write dictionary to disk
    :param dict_to_save: serializable dictionary to save
    :param filepath: path where to save
    :return: void
    """
    with open(str(filepath), 'w') as f:
        json.dump(dict_to_save, f)

def bytesio_to_dict(bytesio_obj):
    # Read the content from the BytesIO object
    content = bytesio_obj.getvalue()

    # Convert the content to a string
    content_str = content.decode('utf-8')

    # Parse the string as a JSON object to convert it to a Python dictionary
    data_dict = json.loads(content_str)

    return data_dict

def get_python_path():
    python_path = sys.executable
    return python_path
