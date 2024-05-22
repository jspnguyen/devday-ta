# %%
import json
import os

# %%
folder = '70files/notes'

# %%
for filename in os.listdir(folder):
    #file_path = os.path.join(folder, filename)

    data = {
        'metadataAttributes': {
            'type': 'note',
            'course': 'cs70',
            'semester': 'sp24'
        }

    }

    metadata_filepath = os.path.join(folder, 'metadata', filename + '.metadata.json')
    with open(metadata_filepath, 'w') as file:
        json.dump(data, file, indent=4)

        file.close()

# %%
