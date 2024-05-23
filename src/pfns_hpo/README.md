# pfns_hpo



## 1. Installation
```
git clone https://github.com/automl/pfns_hpo.git
cd pfns_hpo
conda create -n pfns_hpo python=3.8
conda activate pfns_hpo

# Install for usage
pip install .

# Install for development
make install-dev
```


## 2. Conda, Poetry, Package, Pre-Commit

To setup tooling and install the package, follow this documentation (**removed**) using the environment name of your choice.

**NOTE: Requires Python 3.7**

```bash
poetry install
```

## 3. Just

To install our command runner just you can check the [just documentation](https://github.com/casey/just#installation), or run the below command

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/.just
```

Also, to make the `just` command available you should add

```bash
export PATH="$HOME/.just:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply run the export manually.

## Documentation

Documentation at https://automl.github.io/pfns_hpo/main

