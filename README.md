# nn

## Reading ROOT files into python DataFrames

This is useful to convert the data stored in root files (provided my CERN OpenData) to train the network with

What needs to be done to make the script work: 
1. You have ROOT, otherwise install it `https://root.cern.ch/downloading-root` (adviced to use binaries, link under `For each of the three active versions`)
2. Create a virtual enviroment to make it less messy, make sure it is python 2.7 (the pyroot library is by default often compiled for 2.7)
  * `virtualenv -p <path/to/python2.7> env`
  * `source env/bin/activate`
3. Install pandas and root_pandas, make sure you are using pip with python 2.7
  * `pip --version` => `<path> (python on 2.7)`
  * `pip install pandas`
  * `pip install root_pandas`
