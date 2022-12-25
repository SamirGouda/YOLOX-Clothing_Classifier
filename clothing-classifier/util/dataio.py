import collections
import csv
import pickle
import logging
import torch
import torchaudio
import re
import json
from torch.utils.data import Dataset
import copy
import contextlib
from types import MethodType
import os

logger = logging.getLogger(__name__)

TORCHAUDIO_FORMATS = ["wav", "flac", "aac", "ogg", "flac", "mp3"]
ITEM_POSTFIX = "_data"

CSVItem = collections.namedtuple("CSVItem", ["data", "format", "opts"])
CSVItem.__doc__ = """The Legacy Extended CSV Data item triplet"""

def load_data_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values.

    Arguments
    ----------
    json_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}.
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        JSON data with replacements applied.

    Example
    -------
    >>> json_spec = '''{
    ...   "ex1": {"files": ["{ROOT}/mic1/ex1.wav", "{ROOT}/mic2/ex1.wav"], "id": 1},
    ...   "ex2": {"files": [{"spk1": "{ROOT}/ex2.wav"}, {"spk2": "{ROOT}/ex2.wav"}], "id": 2}
    ... }
    ... '''
    >>> tmpfile = getfixture('tmpdir') / "test.json"
    >>> with open(tmpfile, "w") as fo:
    ...     _ = fo.write(json_spec)
    >>> data = load_data_json(tmpfile, {"ROOT": "/home"})
    >>> data["ex1"]["files"][0]
    '/home/mic1/ex1.wav'
    >>> data["ex2"]["files"][1]["spk2"]
    '/home/ex2.wav'

    """
    with open(json_path, "r") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json

def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    Uses the SpeechBrain legacy CSV data format, where the CSV must have an
    'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), e.g., {"data_folder": "/home/speechbrain/data"}
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.

    Example
    -------
    >>> csv_spec = '''ID,duration,wav_path
    ... utt1,1.45,$data_folder/utt1.wav
    ... utt2,2.0,$data_folder/utt2.wav
    ... '''
    >>> tmpfile = getfixture("tmpdir") / "test.csv"
    >>> with open(tmpfile, "w") as fo:
    ...     _ = fo.write(csv_spec)
    >>> data = load_data_csv(tmpfile, {"data_folder": "/home"})
    >>> data["utt1"]["wav_path"]
    '/home/utt1.wav'
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result

def _recursive_format(data, replacements):
    # Data: dict or list, replacements : dict
    # Replaces string keys in replacements by their values
    # at all levels of data (in str values)
    # Works in-place.
    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[key] = item.format_map(replacements)
            # If not dict, list or str, do nothing
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[i] = item.format_map(replacements)
            # If not dict, list or str, do nothing

def load_sb_extended_csv(csv_path, replacements={}):
    """Loads SB Extended CSV and formats string values.

    Uses the SpeechBrain Extended CSV data format, where the
    CSV must have an 'ID' and 'duration' fields.

    The rest of the fields come in triplets:
    ``<name>, <name>_format, <name>_opts``.

    These add a <name>_sb_data item in the dict. Additionally, a
    basic DynamicItem (see DynamicItemDataset) is created, which
    loads the _sb_data item.

    Bash-like string replacements with $to_replace are supported.

    This format has its restriction, but they allow some tasks to
    have loading specified by the CSV.

    Arguments
    ----------
    csv_path : str
        Path to the CSV file.
    replacements : dict
        Optional dict:
        e.g. ``{"data_folder": "/home/speechbrain/data"}``
        This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    list
        List of DynamicItems to add in DynamicItemDataset.

    """
    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        if not reader.fieldnames[0] == "ID":
            raise KeyError(
                "CSV has to have an 'ID' field, with unique ids"
                " for all data points"
            )
        if not reader.fieldnames[1] == "duration":
            raise KeyError(
                "CSV has to have an 'duration' field, "
                "with the length of the data point in seconds."
            )
        if not len(reader.fieldnames[2:]) % 3 == 0:
            raise ValueError(
                "All named fields must have 3 entries: "
                "<name>, <name>_format, <name>_opts"
            )
        names = reader.fieldnames[2::3]
        for row in reader:
            # Make a triplet for each name
            data_point = {}
            # ID:
            data_id = row["ID"]
            del row["ID"]  # This is used as a key in result, instead.
            # Duration:
            data_point["duration"] = float(row["duration"])
            del row["duration"]  # This is handled specially.
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            # Only need to run these in the actual data,
            # not in _opts, _format
            for key, value in list(row.items())[::3]:
                try:
                    row[key] = variable_finder.sub(
                        lambda match: replacements[match[1]], value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            for i, name in enumerate(names):
                triplet = CSVItem(*list(row.values())[i * 3 : i * 3 + 3])
                data_point[name + ITEM_POSTFIX] = triplet
            result[data_id] = data_point
        # Make a DynamicItem for each CSV entry
        # _read_csv_item delegates reading to further
        dynamic_items_to_add = []
        for name in names:
            di = {
                "func": _read_csv_item,
                "takes": name + ITEM_POSTFIX,
                "provides": name,
            }
            dynamic_items_to_add.append(di)
        return result, dynamic_items_to_add, names


def _read_csv_item(item):
    """Reads the different formats supported in SB Extended CSV.

    Delegates to the relevant functions.
    """
    opts = _parse_csv_item_opts(item.opts)
    if item.format in TORCHAUDIO_FORMATS:
        audio, _ = torchaudio.load(item.data)
        return audio.squeeze(0)
    elif item.format == "pkl":
        return read_pkl(item.data, opts)
    elif item.format == "string":
        # Just implement string reading here.
        # NOTE: No longer supporting
        # lab2ind mapping like before.
        # Try decoding string
        string = item.data
        try:
            string = string.decode("utf-8")
        except AttributeError:
            pass
        # Splitting elements with ' '
        string = string.split(" ")
        return string
    else:
        raise TypeError(f"Don't know how to read {item.format}")


def _parse_csv_item_opts(entry):
    """Parse the _opts field in a SB Extended CSV item."""
    # Accepting even slightly weirdly formatted entries:
    entry = entry.strip()
    if len(entry) == 0:
        return {}
    opts = {}
    for opt in entry.split(" "):
        opt_name, opt_val = opt.split(":")
        opts[opt_name] = opt_val
    return opts


def read_pkl(file, data_options={}, lab2ind=None):
    """This function reads tensors store in pkl format.

    Arguments
    ---------
    file : str
        The path to file to read.
    data_options : dict, optional
        A dictionary containing options for the reader.
    lab2ind : dict, optional
        Mapping from label to integer indices.

    Returns
    -------
    numpy.array
        The array containing the read signal.
    """

    # Trying to read data
    try:
        with open(file, "rb") as f:
            pkl_element = pickle.load(f)
    except pickle.UnpicklingError:
        err_msg = "cannot read the pkl file %s" % (file)
        raise ValueError(err_msg)

    type_ok = False

    if isinstance(pkl_element, list):

        if isinstance(pkl_element[0], float):
            tensor = torch.FloatTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], int):
            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if isinstance(pkl_element[0], str):

            # convert string to integer as specified in self.label_dict
            if lab2ind is not None:
                for index, val in enumerate(pkl_element):
                    pkl_element[index] = lab2ind[val]

            tensor = torch.LongTensor(pkl_element)
            type_ok = True

        if not (type_ok):
            err_msg = (
                "The pkl file %s can only contain list of integers, "
                "floats, or strings. Got %s"
            ) % (file, type(pkl_element[0]))
            raise ValueError(err_msg)
    else:
        tensor = pkl_element

    tensor_type = tensor.dtype

    # Conversion to 32 bit (if needed)
    if tensor_type == "float64":
        tensor = tensor.astype("float32")

    if tensor_type == "int64":
        tensor = tensor.astype("int32")

    return tensor


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Example
    -------
    >>> get_all_files('samples/rir_samples', match_and=['3.wav'])
    ['samples/rir_samples/rir3.wav']
    """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles