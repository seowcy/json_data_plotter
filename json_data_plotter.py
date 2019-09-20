import json, os, re, sys, collections
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from IPython.display import display
from IPython.display import clear_output
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option('display.max_colwidth', -1)

FLATTENED_KEYS = ["submission_filename", "submission_date", 
                  "submission_submitter_country", "submission_submitter_region",
                  "additional_info_magic", "positives"]
DATE_TIME_FIELDS = ["submission_date"]
X_AXIS = "submission_date"
Y_AXIS = "positives"
MD5_PATTERN = r'[A-Fa-f0-9]{32}'
MD5_MASK = re.compile(MD5_PATTERN)
DEPENDENT_FIELDS = ["submission_submitter_country", "submission_submitter_region"]
FUZZ_FIELD = "submission_filename"


def get_list_of_json_files(folder_path):
    """
    Get list of JSON files from a specified folder.
    Returns a list object of filenames.
    """
    return [filename for filename in os.listdir(folder_path) if filename.lower().endswith('.json')]

def flatten(d, parent_key='', sep='_'):
    """
    Takes a nested dictionary and flattens it to just one layer.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_clean(json_dict):
    """
    Returns if the json_dict object meets a few specified criteria.
    """
    filename = json_dict["submission_filename"]
    if MD5_MASK.match(filename):
        return False
    if filename.lower().endswith(".virus") or filename.lower().endswith(".vir"):
        return False
    return True

def get_flat_dict_from_json_file(json_filename, folder_path='.', pprint=False):
    """
    Takes a json_filename and returns a flattened dictionary object by default.
    Alternatively, prints a formatted string of the json file as a dictionary object.
    """
    with open('\\'.join([folder_path, json_filename]), 'r') as f:
        raw = flatten(json.load(f))
    if pprint:
        print(json.dumps(raw, indent=4))
        return
    else:
        return raw

def create_df_from_json_files(folder_path='.', json_filename_list=None, clean=True):
    """
    Returns a Pandas DataFrame object from a list of JSON files.
    Can take a folder path or a list of json_filenames.
    """
    pre_df = {key:[] for key in FLATTENED_KEYS}
    pre_df["_hash"] = []
    if json_filename_list is None:
        json_filename_list = get_list_of_json_files(folder_path)
    for json_filename in json_filename_list:
        raw = get_flat_dict_from_json_file(json_filename, folder_path, False)
        if clean:
            if is_clean(raw):
                pre_df["_hash"].append(json_filename.split('.')[0])
                for key in FLATTENED_KEYS:
                    try:
                        pre_df[key].append(raw[key])
                    except KeyError:
                        pre_df[key].append('-')
        else:
            pre_df["_hash"].append(json_filename.split('.')[0])
            for key in FLATTENED_KEYS:
                try:
                    pre_df[key].append(raw[key])
                except KeyError:
                    pre_df[key].append('-')
    df = pd.DataFrame(pre_df)
    return df

def cluster_data(df, dependent_fields, fields_of_interest=None):
    """
    Takes a Pandas DataFrame as input, and filters the df by a list of dependent fields.
    Returns the fields_of_interest. Returns all fields if None. (Default: None)
    """
    indices = df.groupby(dependent_fields).size().index
    for index in indices:
        query = " and ".join(["%s == '%s'" % (dependent_fields[i], index[i]) for i in range(len(dependent_fields))])
        yield df.query(query).reset_index(drop=True)

def get_field_similarities(cluster, field, threshold=80):
    """
    Creates a generator object that yields similar rows in a cluster.
    """
    data = cluster[field].values
    for i in range(len(data)):
        similarities = [data[i]]
        others = list(data[0:i]) + list(data[i+1:len(data)])
        for other in others:
            similarity_score = fuzz.token_sort_ratio(data[i],other)
            if similarity_score > threshold:
                similarities.append(other)
        if len(similarities) > 2:
            yield cluster[cluster[field].isin(similarities)]
        
def main(folder_path):
    df = create_df_from_json_files(folder_path)
    clusters = cluster_data(df, DEPENDENT_FIELDS)
    for cluster in clusters:
        result_df_generator = get_field_similarities(cluster, FUZZ_FIELD)
        for result_df in result_df_generator:
            for date_time_field in DATE_TIME_FIELDS:
                result_df[date_time_field] = pd.to_datetime(result_df[date_time_field])
            result_df = result_df.sort_values(by=DATE_TIME_FIELDS)
            result_df.plot(X_AXIS, Y_AXIS, ms=10, style="-o", figsize=(15,5))
            plt.show()
            display(result_df)
            dump = input("Press ENTER to continue.")
            clear_output()
    print("END")
    return
    

if __name__ == '__main__':
    args = sys.argv[1:]
    program = os.path.basename(sys.argv[0])
    if len(args) != 1:
        print("Usage: %s <folder_path>" % program)
    else:
        main(args[0])

