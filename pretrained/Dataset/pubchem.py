# pubchem web:https://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/10_conf_per_cmpd/SDF/
# There are 1,0000,0000 molecules, which one has 10 confomers, so there are 1 billion structures

import requests
import gzip
import os

def download_and_decompress(id):
    # Define the directory to save the downloaded file
    url = f"https://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/10_conf_per_cmpd/SDF/{id}.sdf.gz"
    directory = "sdf/"

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract filename from the URL
        filename = url.split("/")[-1]

        # Path to save the downloaded file
        file_path = os.path.join(directory, filename)

        # Open the file and write the content
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded successfully to: {file_path}")

        # If the downloaded file is in gzip format, decompress it
        if filename.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f_in:
                # Read the decompressed data
                decompressed_data = f_in.read()

            # Define the path for the decompressed file
            decompressed_file_path = os.path.splitext(file_path)[0]

            # Write the decompressed data to a new file
            with open(decompressed_file_path, 'wb') as f_out:
                f_out.write(decompressed_data)

            print(f"Decompressed file saved successfully: {decompressed_file_path}")
    else:
        print("Failed to download the file.")


def generate_sequence(term, n):
    sequence = []
    for i in range(n):
        formatted_term = f"{term:08}"
        sequence.append(formatted_term)
        term += 25000
    return sequence

def main():
    # Let's download 100,0000 molecules but with 10X structures.
    fs=generate_sequence(1, 40)
    bs=generate_sequence(25000, 40)
    for i in range(len(fs)):
        id = fs[i] + "_" + bs[i]
        download_and_decompress(id)

if __name__ == "__main__":
    main()