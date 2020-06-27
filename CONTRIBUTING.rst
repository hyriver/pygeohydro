.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/cheginit/hydrodata/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Other than new features that you might have in mind, you can look through
the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Hydrodata could always use more documentation, whether as part of the
official Hydrodata docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/cheginit/hydrodata/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `hydrodata` for local development.

1. Fork the `hydrodata` repo on GitHub.
2. Clone your fork locally:

.. code-block:: console

    $ git clone git@github.com:your_name_here/hydrodata.git

3. Install your local copy into a virtualenv. Assuming you have conda installed, this is how you set up your fork for local development:

.. code-block:: console

    $ cd hydrodata/
    $ conda env create -f ci/requirements/environment.yml
    $ conda activate hydrodata-dev
    $ python -m pip install . --no-deps

4. Create a branch for local development:

.. code-block:: console

    $ git checkout -b name-of-your-bugfix-or-feature

5. Before you first commit, pre-commit hooks needs to be setup:

.. code-block:: console

    $ pre-commit install
    $ pre-commit run --all-files

6. Now you can make your changes locally, make sure to add a discription of the changes to ``HISTORY.rst`` file and add extra tests, if applicable, to ``tests`` folder. Afterward, you can install and test the code:

.. code-block:: console

    $ make clean
    $ make lint
    $ make install
    $ make coverage

7. Commit your changes and push your branch to GitHub:

.. code-block:: console

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6, 3.7 and 3.8. Check
   https://github.com/cheginit/hydrodata/actions
   and make sure that the tests pass for all supported Python versions.
4. Whenever you add an item to ``HISTORY.rst`` file make sure to add your name
   at the end of the item like this ``By `Taher Chegini <https://github.com/cheginit>`_``

Tips
----

To run a subset of tests:

.. code-block:: console

    $ pytest -k "test_name1 or test_name2"

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run:

.. code-block:: console

    $ bump2version patch # possible: major / minor / patch
    $ git push
    $ git push --follow-tags

Then release the tag and Github Actions will deploy it to PyPi.
