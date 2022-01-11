import glob
import os.path

folder_name = 'F'
my_path = os.path.join('bonn',folder_name)
read_files = glob.glob(my_path + "/*.txt")
print(read_files)
with open("bonn_" + folder_name + ".txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
