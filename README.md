hunsberger-neco2014
===================

Source code for "The competing benefits of noise and heterogeneity in neural
coding" by Eric Hunsberger, Matthew Scott, and Chris Eliasmith. The manuscript
included in this repository is a preprint of the final article to be
published in Neural Computation (see
<http://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00621>).

All the code in this repository has been run on a Linux machine. While ideally
it should work without modification on other machines, if you are having
trouble getting the code to run, or have any other questions relating to the
paper or the code, please contact Eric Hunsberger at <ehunsber@uwaterloo.ca>.


Requirements
------------

The core requirement for this repository is Python 2.7. Python requirements
can be found in `requirements.txt`, and can be installed using `pip` with

    pip install requirements.txt

However, the two first requirements are Numpy and Scipy, which must be
installed specially on some machines. For instructions specific to your
machine, please visit <http://www.scipy.org/install.html>.

Building the paper requires LaTeX, specifically the existence of
a `pdflatex` command at the command-line.


`doit` usage
------------

All the scripts in this project, namely those for running the simulations,
creating the figures, and putting together the paper, have been set up to
run easily with `doit`. [`doit`](http://www.pydoit.org) is a make-like utility
for managing tasks, and running them only as necessary. The full list of tasks
can be seen by typing

    doit list

into the main console. Each command can then be run by typing `doit` and the
command name. For example, to build the paper, type

    doit paper

The simulation tasks have been set not to run if the target results file
exists at all. To rerun the simulations, manually remove all the results files
(ending in `.npz`) from the `results` directory. Then call

    doit

to rerun all out-of-date tasks (which should be everything). To rerun specific
tasks, such as only the simulations for the information plots, you can use
a wildcard character like

    doit sim_info*

to run `sim_info_lif` and `sim_info_fhn`. Note that some of the simulations may
take a while to run (on the order of one day). To run the simulations faster,
you can ask Theano to run them on your GPU by passing the argument `gpu=true`
to `doit`.

    doit gpu=true

This requires Theano to work with your GPU (see the first paragraphs of
<http://deeplearning.net/software/theano/tutorial/using_gpu.html>).


Scripts
-------

The `scripts` folder contains all the source code for conducting the numerical
experiments, as well as code for generating the plots.
