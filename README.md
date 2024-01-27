# seismic_examples

# Install Conda
Protect your system `python` build system by [installing Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).  


# Setting up local environment
Now that you've got conda setup, run the following series of commands in the terminal.
```
cd /whatever/repo/path
git clone name_of_repo
cd name_of_repo
conda create -n flatvel python=3.10
pip install -r requirements.txt
```

# Getting the data
Simply download the data from the `iomt` UT box folder and put in the directory `/whatever/repo/path`.

# Running the code
Just run
```
cd FlatVel_A
python plt.py
```
and that should generate all the exact same plots that are currently in the UT box `plots` folder.
All the variables that are set for the plotting configuration are set in the file `conf/plots.yaml` file.
This is a somewhat unconventional programming style, but I think it's cleaner since all your variables are on spot rather than scattered throughout the code.
Furthermore, the `helpers.py` file you don't need to worry about much...these are a bunch of functions that make tedious tasks simpler to code with.
The only one that you might need to worry about is the `bool_slice` function. 
This function is a glorified nested `for` loop that allows you to change the order that you loop in easily. 
Let me know if you have any trouble running the code and/or understanding anything.
