### Setting up your development environment manually
Follow these steps to manually install all the libraries needed for the projects in this book. Otherwise, skip to the next section to install the environment created for you in the book’s GitHub repo.
1. On your terminal, create a new conda environment with Python 3 and call it **deep_learning_for_vision_systems**:
    ```bash
    conda create -n deep_learning_for_vision_systems python=3
    ```
   Note that to remove a conda environment, you use ``conda env remove -n <env_name>``.
1. Activate your environment. You must activate the environment before installing your packages. This way, all packages are installed only for this environment:
    ```bash
    conda activate deep_learning_for_vision_systems 
    ```
    Note that to deactivate an environment, you use ``conda deactivate <env_name>``.
    
    Now you are inside your new environment. To see the default packages installed in this environment, type the following command: conda list. Next, you will install the packages used for the projects in this book.
1. Install NumPy, pandas, and Matplotlib. These are very common ML packages that you will almost always use in your projects for math operations, data manipulation, and visualization tasks:
    ```bash
    conda install numpy pandas matplotlib
    ```
     Note that throughout these installs, you will be prompted to confirm to proceed (Proceed ([y]/n)?). Type y and press Enter to continue the installation.
1. Install the Jupyter notebooks. We use Jupyter notebooks in this book for easier development:
    ```bash
    conda install jupyter notebook
    ```
1. Install OpenCV (the most popular open source CV library):
    ```bash
    conda install -c conda-forge opencv
    ```
1. Install Keras:
    ```bash
    pip install keras
    ```
1. Install TensorFlow:
    ```bash
    pip install tensorflow
    ```
    Now everything is complete and your environment is ready to start developing. If you want to view all the libraries installed in your environment, type the following command:
    ```bash
    conda list
    ```
    These packages are separate from your other environments. This way, you can avoid any version-conflict issues.
### Using the conda environment in the book’s repo
1. Clone the book’s GitHub repository from https://github.com/moelgendy/deep_learning_for_vision_systems. The environment is located in the installer/application.yaml file:
    ```bash
    cd installer
    ```
1. Create the conda deep_learning environment:
    ```bash
    conda env create -f my_environment.yaml
    ```
1. Activate the conda environment:
    ```bash
    conda activate deep_learning
    ```
1. Launch your Jupyter notebook (make sure you are located in the root of the deep_learning_for_vision_systems repository):
    ```bash
    jupyter notebook
    ```
Now you are ready to run the notebooks associated with the book.

###  Saving and loading environments
It is best practice to save your environment if you want to share it with others so that they can install all the packages used in your code with the correct version. To do that, you can save the packages to a YAML (https://yaml.org) file with this command:
```bash
conda env export > my_environment.yaml 
```
This way, others can use this YAML file to replicate your environment on their machine using the following command:
```
conda env create -f my_environment.yaml 
```
You can also export the list of packages in an environment to a .txt file and then include that file with your code. This allows other people to easily load all the dependencies for your code. Pip has similar functionality with this command:
```
pip freeze > requirements.txt
```
You can find the environment details used for this book’s projects in the downloaded code in the installer directory. You can use it to replicate my environment in your machine.
# Using Docker
## Building Dockerfiles

The image is about 7.83 ~ 8.12 GB.
```bash
docker build -t jupyter .
```

You can add mount parameters to the run command with the -v option. This is useful if you want the docker container to share directories with the host machine. Otherwise, you can copy files using the docker-copy command
```bash
docker run -it --rm --net=host jupyter
```

## Jupyter Docker Stacks
* https://jupyter-docker-stacks.readthedocs.io/en/latest/
* https://github.com/jupyter/docker-stacks/tree/main/images/scipy-notebook
```
docker run -p 10000:8888 quay.io/jupyter/scipy-notebook:2024-01-15

docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/datascience-notebook:2024-01-15
```

http://10.100.198.102:10000/

We use old image: https://jupyter-docker-stacks.readthedocs.io/en/latest/#using-old-images
* https://github.com/jupyter/docker-stacks/tree/main/images/datascience-notebook
``` 
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/datascience-notebook:b86753318aa1
```

http://10.100.198.102:10000/

# Podman Machine Cli
## for winodws
* Install windows wsl2
```powershell
wsl --install --no-distribution
```
* Install podman-cli
```powershell
choco install -y podman-cli
```
# Recipes provide ways to be more productive in Jupyter and Python
* Obtaining the history of Jupyter commands and outputs
* Auto-reloading packages
* Debugging
  * Timing code execution
  * Displaying progress bars
* Compiling your code
* Speeding up pandas DataFrames
* Parallelizing your code

## Obtaining the history of Jupyter commands and outputs
There are lots of different ways to obtain the code in Jupyter cells programmatically. Apart from these inputs, you can also look at the generated outputs. We'll get to both, and we can use global variables for this purpose.

### Execution history
In order to get the execution history of your cells, the ``_ih`` list holds the code of executed cells. In order to get the complete execution history and write it to a file, you can do the following:
```python
with open('command_history.py', 'w') as file:
    for cell_input in _ih[:-1]:
        file.write(cell_input + '\n')
```

If up to this point, we only ran a single cell consisting of ``print('hello, world!')``, that's exactly what we should see in our newly created file, ``command_history.py``:
```python
!cat command_history.py
print('hello, world!')
```
On Windows, to print the content of a file, you can use the ``type`` command.

Instead of ``_ih``, we can use a shorthand for the content of the last three cells.``_i`` gives you the code of the cell that just executed, ``_ii`` is used for the code of the cell executed before that, and ``_iii`` for the one before that.

### Outputs
In order to get recent outputs, you can use ``_`` (single underscore), ``__`` (double underscore), and ``___`` (triple underscore), respectively, for the most recent, second, and third most recent outputs.
## Auto-reloading packages
``autoreload`` is a built-in extension that reloads the module when you make changes to a module on disk. It will automagically reload the module once you've saved it. 

Instead of manually reloading your package or restarting the notebook, with ``autoreload``, the only thing you have to do is to load and enable the extension, and it will do its magic.

We first load the extension as follows:
```python 
%load_ext autoreload
```
And then we enable it as follows:
```python
%autoreload 2
```

This can save a lot of time when you are developing (and testing) a library or module. 
## Debugging
If you cannot spot an error and the traceback of the error is not enough to find the problem, debugging can speed up the error-searching process a lot. Let's have a quick look at the debug magic:
1. Put the following code into a cell:
   ```python
   def normalize(x, norm=10.0):
     return x / norm

   normalize(5, 1)
   ```
   You should see 5.0 as the cell output.

   However, there's an error in the function, and I am sure the attentive reader will already have spotted it. Let's debug!
2. Put this into a new cell:
   ```python
   %debug
   normalize(5, 0)
   ```
3. Execute the cell by pressing ``Ctrl + Enter`` or ``Alt + Enter``. You will get a debug prompt:
   ```python
   > <iPython-input-11-a940a356f993>(2)normalize() 
        1 def normalize(x, norm=10): ----> 
        2   return x / norm 
        3 
        4 normalize(5, 1) 
   ipdb> a 
   x = 5 
   norm = 0 
   ipdb> q
   --------------------------------------------------------------------------- ZeroDivisionError                         Traceback (most recent call last)
   <iPython-input-13-8ade44ebcb0c> in <module>()
        1 get_iPython().magic('debug') ---->
        2 normalize(5, 0)


   <iPython-input-11-a940a356f993> in normalize(a, norm)
        1 def normalize(x, norm=10): ----> 
        2   return x / norm 
        3 
        4 normalize(5, 1) 
   ZeroDivisionError: division by zero
   ```
   We've used the argument command to print out the arguments of the executed function, and then we quit the debugger with the quit command. You can find more commands on **The Python Debugger (pdb)** documentation page at https://docs.Python.org/3/library/pdb.html.

   Let's look at a few more useful magic commands. 
### Timing code execution
Once your code does what it's supposed to, you often get into squeezing every bit of performance out of your models or algorithms. For this, you'll check execution times and create benchmarks using them. Let's see how to time executions.

There is a built-in magic command for timing cell execution – ``timeit``. The ``timeit`` functionality is part of the Python standard library (https://docs.Python.org/3/library/timeit.html). It runs a command 10,000 times (by default) in a period of 5 times inside a loop (by default) and shows an average execution time as a result:
```python
%%timeit -n 10 -r 1
import time
time.sleep(1)
```
We see the following output:
```python
1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each)
```
The ``iPython-autotime`` library (https://github.com/cpcloud/iPython-autotime) is an external extension that provides you the  timings for all the cells that execute, rather than having to use %%timeit every time:

1. Install ``autotime`` as follows:
   ```python
   pip install iPython-autotime
   ```
   Please note that this syntax works for Colab, but not in standard Jupyter Notebook. What always works to install libraries is using the pip or conda magic commands, ``%pip`` and ``%conda``, respectively. Also, you can  execute any  shell command from the notebook if you start your line with an exclamation mark, like this:
   ```python
   !pip install iPython-autotime
   ```
2. Now let's use it, as follows:
   ```python
   %load_ext autotime
   ```
   Test how long a simple list comprehension takes with the following command:
   ```python
   sum([i for i in range(10)])
   ```
   We'll see this output: 
   ```bash
   time: 5.62 ms.
   ```

Hopefully, you can see how this can come in handy for comparing different implementations. Especially in situations where you have a lot of data, or complex processing, this can be very useful.
### Displaying progress bars
Even if your code is optimized, it's good to know if it's going to finish in minutes, hours, or days. ``tqdm`` provides progress bars with time estimates. If you aren't sure how long your job will run, it's just one letter away – in many cases, it's just a matter of changing ``range`` for ``trange``:
   ```python
   from tqdm.notebook import trange
   from tqdm.notebook import tqdm
   tqdm.pandas()
   ```
The ``tqdm`` pandas integration (optional) means that you can see progress bars for pandas ``apply`` operations. Just swap ``apply`` for ``progress_apply``.

For Python loops just wrap your loop with a tqdm function and voila, there'll be a progress bar and time estimates for your loop completion!
```python
global_sum = 0.0
for i in trange(1000000):
   global_sum += 1.0
```
Tqdm provides different ways to do this, and they all require minimal code changes - sometimes as little as one letter, as you can see in the previous example. The more general syntax is wrapping your loop iterator with tqdm like this:
```python
for _ in tqdm(range(10)):
   print()
```
So, next time you are just about to set off long-running loop, and you are not just sure how long it will take, just remember this sub-recipe, and use ``tqdm``.

## Compiling your code
Python is an interpreted language, which is a great advantage for experimenting, but it can be detrimental to speed. There are different ways to compile your Python code, or to use compiled code from Python.

Let's first look at Cython. Cython is an optimizing static compiler for Python, and the programming language compiled by the Cython compiler. The main idea is to write code in a language very similar to Python, and generate C code. This C code can then be compiled as a binary Python extension. SciPy (and NumPy), scikit-learn, and many other libraries have significant parts written in Cython for speed up. You can find out more about Cython on its website at https://cython.org/:

1. You can use the Cython extension for building cython functions in your notebook:
   ```python
   %load_ext Cython
   ```
2. After loading the extension, annotate your cell as follows:
   ```python
   %%cython
   def multiply(float x, float y):
       return x * y
   ```
3. We can call this function just like any Python function – with the added benefit that it's already compiled:
   ```python
   multiply(10, 5)  # 50
   ```
   This is perhaps not the most useful example of compiling code. For such a small function, the overhead of compilation is too big. You would probably want to compile something that's a bit more complex. 

   Numba is a JIT compiler for Python (https://numba.pydata.org/). You can often get a speed-up similar to C or Cython using numba and writing idiomatic Python code like the following:
   ```python
   from numba import jit
   @jit
   def add_numbers(N):
       a = 0
       for i in range(N):
           a += i
   add_numbers(10)           
   ```
   With autotime activated, you should see something like this: 
   ```python
   time: 2.19 s          
   ```
   So again, the overhead of the compilation is too big to make a meaningful impact. Of course, we'd only see the benefit if it's offset against the compilation. However, if we use this function again, we should see a speedup. Try it out yourself! Once the code is already compiled, the time significantly improves:
   ```python
   add_numbers(10)      
   ```
   You should see something like this:
   ```python
   time: 867 µs    
   ```
   There are other libraries that provide JIT compilation including TensorFlow, PyTorch, and JAX, that can help you get similar benefits.

   The following example comes directly from the JAX documentation, at https://jax.readthedocs.io/en/latest/index.html:
   ```python
   import jax.numpy as np
   from jax import jit
   def slow_f(x):
       return x * x + x * 2.0

   x = np.ones((5000, 5000)) 
   fast_f = jit(slow_f) 
   fast_f(x)    
   ```
So there are different ways to get speed benefits from using JIT or ahead-of-time compilation. We'll see some other ways of speeding up your code in the following sections.
## Speeding up pandas DataFrames
One of the most important libraries throughout this book will be ``pandas``, a library for tabular data that's useful for **Extract, Transform, Load (ETL)** jobs. Pandas is a wonderful library, however; once you get to more demanding tasks, you'll hit some of its limitations. Pandas is the go-to library for loading and transforming data. One problem with data processing is that it can be slow, even if you vectorize the function or if you use ``df.apply()``.

You can move further by parallelizing ``apply``. Some libraries, such as ``swifter``, can help you by choosing backends for computations for you, or you can make the choice yourself:

* You can use Dask DataFrames instead of pandas if you want to run on multiple cores of the same or several machines over a network.
* You can use CuPy or cuDF if you want to run computations on the GPU instead of the CPU. These have stable integrations with Dask, so you can run both on multiple cores and multiple GPUs, and you can still rely on a pandas-like syntax (see https://docs.dask.org/en/latest/gpu.html).

As we've mentioned, ``swifter`` can choose a backend for you with no change of syntax. Here is a quick setup for using ``pandas`` with ``swifter``:
```python
mport pandas as pd
import swifter

df = pd.read_csv('some_big_dataset.csv')
df['datacol'] = df['datacol'].swifter.apply(some_long_running_function)
```
Generally, apply() is much faster than looping over DataFrames.

You can further improve the speed of execution by using the underlying NumPy arrays directly and accessing NumPy functions, for example, using ``df.values.apply()``. NumPy vectorization can be a breeze, really. See the following example of applying a NumPy vectorization on a pandas DataFrame column:
```python
squarer = lambda t: t ** 2
vfunc = np.vectorize(squarer)
df['squared'] = vfunc(df[col].values)
```
These are just two ways, but if you look at the next sub-recipe, you should be able to write a parallel map function as yet another alternative.
## Parallelizing your code
One way to get something done more quickly is to do multiple things at once. There are different ways to implement your routines or algorithms with parallelism. Python has a lot of libraries that support this functionality. Let's see a few examples with multiprocessing, Ray, joblib, and how to make use of scikit-learn's parallelism.

The multiprocessing library comes as part of Python's standard library. Let's look at it first. We don't provide a dataset of millions of points here – the point is to show a usage pattern – however, please imagine a large dataset. Here's a code snippet of using our pseudo-dataset:
```python
# run on multiple cores
import multiprocessing

dataset = [
    {
        'data': 'large arrays and pandas DataFrames',
        'filename': 'path/to/files/image_1.png'
    }, # ... 100,000 datapoints
]

def get_filename(datapoint):
    return datapoint['filename'].split('/')[-1]

pool = multiprocessing.Pool(64)
result = pool.map(get_filename, dataset)
```
Using Ray, you can parallelize over multiple machines in addition to multiple cores, leaving your code virtually unchanged. Ray efficiently handles data through shared memory (and zero-copy serialization) and uses a distributed task scheduler with fault tolerance:
```python
# run on multiple machines and their cores
import ray
ray.init(ignore_reinit_error=True)

@ray.remote
def get_filename(datapoint):
    return datapoint['filename'].split('/')[-1]

result = []
for datapoint in dataset:
    result.append(get_filename.remote(datapoint))
```
Scikit-learn, the machine learning library we installed earlier, internally uses joblib for parallelization. The following is an example of this:

```python
from joblib import Parallel, delayed

def complex_function(x):
    '''this is an example for a function that potentially coult take very long.
    '''
    return sqrt(x)

Parallel(n_jobs=2)(delayed(complex_function)(i ** 2) for i in range(10))
```
This would give you [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]. We took this example from the joblib examples about parallel for loops, available at https://joblib.readthedocs.io/en/latest/parallel.html.

When using scikit-learn, watch out for functions that have an n_jobs parameter. This parameter is directly handed over to joblib.Parallel (https://github.com/joblib/joblib/blob/master/joblib/parallel.py). none (the default setting) means sequential execution, in other words, no parallelism. So if you want to execute code in parallel, make sure to set this n_jobs parameter, for example, to -1 in order to make full use of all your CPUs.

### Setting up your development environment manually
Follow these steps to manually install all the libraries needed for the projects in this book. Otherwise, skip to the next section to install the environment created for you in the book’s GitHub repo.
1. On your terminal, create a new conda environment with Python 3 and call it **deep_learning_for_vision_systems**:
    ```bash
    conda create -n deep_learning_for_vision_systems python=3
    ```
   Note that to remove a conda environment, you use ``conda env remove -n <env_name>``.
1. Activate your environment. You must activate the environment before installing your packages. This way, all packages are installed only for this environment:
    ```bash
    conda activate deep_learning_for_vision_systems 
    ```
    Note that to deactivate an environment, you use ``conda deactivate <env_name>``.
    
    Now you are inside your new environment. To see the default packages installed in this environment, type the following command: conda list. Next, you will install the packages used for the projects in this book.
1. Install NumPy, pandas, and Matplotlib. These are very common ML packages that you will almost always use in your projects for math operations, data manipulation, and visualization tasks:
    ```bash
    conda install numpy pandas matplotlib
    ```
     Note that throughout these installs, you will be prompted to confirm to proceed (Proceed ([y]/n)?). Type y and press Enter to continue the installation.
1. Install the Jupyter notebooks. We use Jupyter notebooks in this book for easier development:
    ```bash
    conda install jupyter notebook
    ```
1. Install OpenCV (the most popular open source CV library):
    ```bash
    conda install -c conda-forge opencv
    ```
1. Install Keras:
    ```bash
    pip install keras
    ```
1. Install TensorFlow:
    ```bash
    pip install tensorflow
    ```
    Now everything is complete and your environment is ready to start developing. If you want to view all the libraries installed in your environment, type the following command:
    ```bash
    conda list
    ```
    These packages are separate from your other environments. This way, you can avoid any version-conflict issues.
### Using the conda environment in the book’s repo
1. Clone the book’s GitHub repository from https://github.com/moelgendy/deep_learning_for_vision_systems. The environment is located in the installer/application.yaml file:
    ```bash
    cd installer
    ```
1. Create the conda deep_learning environment:
    ```bash
    conda env create -f my_environment.yaml
    ```
1. Activate the conda environment:
    ```bash
    conda activate deep_learning
    ```
1. Launch your Jupyter notebook (make sure you are located in the root of the deep_learning_for_vision_systems repository):
    ```bash
    jupyter notebook
    ```
Now you are ready to run the notebooks associated with the book.

###  Saving and loading environments
It is best practice to save your environment if you want to share it with others so that they can install all the packages used in your code with the correct version. To do that, you can save the packages to a YAML (https://yaml.org) file with this command:
```bash
conda env export > my_environment.yaml 
```
This way, others can use this YAML file to replicate your environment on their machine using the following command:
```
conda env create -f my_environment.yaml 
```
You can also export the list of packages in an environment to a .txt file and then include that file with your code. This allows other people to easily load all the dependencies for your code. Pip has similar functionality with this command:
```
pip freeze > requirements.txt
```
You can find the environment details used for this book’s projects in the downloaded code in the installer directory. You can use it to replicate my environment in your machine.