

How to use this example
=======================

1. Create a virtual environment::

    $ virtualenv keras-env
    $ source keras-env/bin/activate


2. Install the prerequisites::

    $ pip install -r requirements.txt


3. Generate the datasets::
   
   $ python 00_generate_wine_dataset.py


4. Run the example::
   
   $ python 4_wine_net_fit_generator_sql.py
