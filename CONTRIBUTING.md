# Contributing to Temporal

First off, thank you for considering contributing to Temporal! It's people like you that make the open source community such a great community! ❤️

We welcome any type of contribution, not only code. You can help with 
- **QA**: file bug reports, the more details you can give the better (e.g. platform, versions, stack traces)
- **Marketing**: writing blog posts, how-to's, etc.
- **Community**: presenting the project at meetups, organizing a workshop, etc.
- **Code**: take a look at the open issues. Even if you can't write code, commenting on them, showing that you care about a given issue matters. It helps us triage them.
- **Money**: we welcome financial contributions in full transparency on our [open collective](https://opencollective.com/temporal).

## Developing for Temporal

To get started with developing for Temporal, you will need to have `poetry` installed. You can find instructions on how to install it on the [official website](https://python-poetry.org/docs/).

Once you have `poetry` installed, you can clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/temporal.git
cd temporal
poetry install
```

This will create a virtual environment with all the necessary dependencies to run, test, and develop Temporal.

### Running Tests

To run the tests, you can use the following command:

```bash
poetry run pytest
```

### Submitting a Pull Request

Before submitting a pull request, please make sure that you have run the tests and that they are all passing.

When you are ready to submit your pull request, please make sure to:

- Provide a clear and descriptive title for your pull request.
- Provide a detailed description of the changes you have made.
- If your pull request is related to an existing issue, please reference the issue in your pull request description.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. You can find the Code of Conduct in the `CODE_OF_CONDUCT.md` file in this repository.
