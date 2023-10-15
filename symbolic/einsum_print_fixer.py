import pathlib as pl 
import re

def read_EinsumPrinted(path):
    string = ""
    with open(path, "r") as file:
        for line in file:
            string += line        

    return string

def find_einsum_args(string):
    # Store einsum strings and the relevant tensors
    keep_next = False
    einsum_strings = []
    einsum_tensors = []
    for i, line in enumerate(string.split("\n")):
        # If the einsum function call is found, the next line will the arguments
        if "einsum" in line:
            keep_next = True
        elif keep_next:
            # Separate tensor index strings and tensors
            
            # Start by getting the string and storing 
            einsum_string, = re.findall(r'"(.*?)"', line)
            indims, outdims = einsum_string.split("->")
            indims = indims.split(",")

            einsum_strings.append(indims + [outdims])
    
            # Then get the tensors
            tensors = re.findall('(?:(?<=, )|^)([^",\s]+)(?=(?:, [^",\s]+)*$)', line.strip())
            einsum_tensors.append(tensors)

            keep_next = False
    
    return einsum_strings, einsum_tensors


def create_tensor_slice(s, t):
    # Makes tensor (t) go from "u" to "u[o, v, o, v]" based on Einsum idicies
    vir = ["a", "b", "c", "d", "e", "f"]
    occ = ["i", "j", "k", "l", "m", "n"]
    vir_slice = "v"
    occ_slice = "o"


    slices = []

    for char in s:
        slice_char = ""
        if char in vir:
            slice_char = vir_slice
        else:
            slice_char = occ_slice

        slices.append(slice_char)

    slices = f"{t}[" + ", ".join(slices) + "]"

    return slices

def construct_old(strings, tensors):
    # Constructs the old einsum arguments based on indicies and tensors
    old = ",".join(strings[:-1])
    old += f"->{strings[-1]}"
    old = '"' + old + '", '

    old += ", ".join(tensors) + "\n"

    return old


def construct_new(strings, tensors, tensor_slice):
    # Constructs the new einsum arguments based on indicies and tensors
    new = ",".join(strings[:-1])
    new += f"->{strings[-1]}"
    new = '"' + new + '", '

    # Add slices to tensors present in "tensor_slice"
    for i, (s, t) in enumerate(zip(strings[:-1], tensors)):
        if t in tensor_slice:
            tensors[i] = create_tensor_slice(s, t)

    # Add optimize
    new += ", ".join(tensors) + ", optimize=True\n"

    return new


def create_einsum_args(einsum_strings, einsum_tensors, tensor_slice):
    old, new = [], []

    for strings, tensors in zip(einsum_strings, einsum_tensors):
        old.append(construct_old(strings, tensors))
        new.append(construct_new(strings, tensors, tensor_slice))

    return old, new

def reformat_EinsumPrinted(string, **kwargs):

    options = {
        "tensor_slice": ["u", "f"],
        "import_as": None,
        "functions": ["einsum", "zeros"],
        "sizes": ["N", "M"],
    }
    options.update(kwargs)

    # Remove superscript from intermediates, underscores from variable names rename lambda
    string = re.sub(r"\^", "", string)
    string = re.sub(r"_", "", string)
    string = re.sub(r"\\lambda", "l", string)

    # Replace old viritual and occupied set sizes with the variable names
    string = string.replace("no", options["sizes"][0], -1)
    string = string.replace("nv", options["sizes"][1], -1)

    # Change del statements to None for deallocation
    string = re.sub(r"del (\w+)", r"\1 = None",string)

    # Find all intermediates, based a var named defined with a specific (4d) size
    intermediates = re.findall(r"(\w+) = zeros\(\((\w+), (\w+), (\w+), (\w+)\)\)", string)

    # Find arguments for each einsum line
    einsum_args = find_einsum_args(string)

    # Replace all functions with import name, defalt to numpy
    if options["import_as"] is not None:
        import_as = options["import_as"]
        for function in options["functions"]:
            string = re.sub(rf"{function}", rf"{import_as}.{function}", string)


    # Parse the arguments set into einsum
    einsum_strings, einsum_tensors = find_einsum_args(string)

    # Create and recreate the new and old arguments
    einsum_old, einsum_new = create_einsum_args(einsum_strings, einsum_tensors, options["tensor_slice"])

    # Swap them all out!
    for old, new in zip(einsum_old, einsum_new):
        string = string.replace(old, new)

    return string


def write_EinsumPrinted(filename, new_string):
    with open(filename, "w+") as file:
        file.write(new_string)

def fix(readpath, writepath):
    path = pl.Path(readpath)

    assert path.exists(), f"The file {str(path)} does not exist"
    assert path.is_file(), f"The object {str(path)} is not a file"

    string = read_EinsumPrinted(path)
    new_string = reformat_EinsumPrinted(string, import_as="np", tensor_slice=["u", "f"])

    write_EinsumPrinted(writepath, new_string)