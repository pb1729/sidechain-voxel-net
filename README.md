# voxel-based protein modelling

Most protein models use graph neural networks (GNNs) to represent the chemical graph. This means that if we are trying to design a protein, a task where we do not yet know the sequence, we need to jointly predict structure and sequence. Many models used for protein design do not do this, forcing user to follow a two stage process and first run a structure model to generate a backbone shape and then use a model like MPNN to generate a sequence conditional on that structure.

Some GNN models do generate both sequence/side-chain positions and the backbone shape jointly. But the methods used to achieve this with a GNN are quite hacky and awkward (add extra atoms whose identity is unknown, force the model to treat them differently based on the kind of amino acid they end up being a part of and place them in a fallback position, such as on top of the alpha carbon, if they are redundant (which must be the case sometimes because of the varying numbers of atoms in different kinds of amino acid).

By directly predicting a density function that encodes the positions of atoms in space, and various other features of the protein, we can handle side-chains cleanly and elegantly. Spatially local interactions become simple convolutions, rather than forming neighbour lists by proximity checking or other methods used in GNNs.

This method is completely analogous to flow-based image generation models, the primary difference being that we are generating a 3 dimensional grid, not a 2 dimensional one.

This has been experimented with here: https://ar5iv.labs.arxiv.org/html/2506.19820v1

In that work, they only generate a protein backbone, which is throwing one of the main advantages of voxel-based generation! We'll generate density functions that allow us to jointly predict structure, sequence, and side-chain positions (all non-hydrogen atoms). Our density function also records chain data like start and end position in the hope that this will force the model to generate chains that are globally sensible, a problem that was observed with the previous work.

# design

Gridpoint spacing is one angstrom. We use an autoencoder to compress grid data to a spacing of 2 angstroms, a reduction of 8x in the number of gridpoints. The autoencoder latents should also more closely approximate a N(0, I) distribution than the raw density values, which is good for flow models.

Initial flownet design will be a convolutional UNet, same as in the paper above. We may have to add more global connectivity depending on our observations of the performance of this initial architecture.

# running

First clone this repo.

## dataset

We use the CATH-S40 dataset, non-redundant version, from here:

https://zenodo.org/records/8388270/files/cath-dataset-nonredundant-S40.pdb.tgz?download=1

Steps: Download and extract. Should produce a directory named `cath-cif`. We want this under the project root directory, so move it there if it's not already there.

## get dependencies

```
mkdir env
python -m venv env
source env/bin/activate
```

Install the version of pytorch that matches your hardware here. Then run:

```
pip install -r requirements.txt
```

to get the rest.

## train the vae

```
mkdir models
python train_vae.py
```

Check what is printed for "device" when the training run starts. It should be something like `cuda:0` unless you're running on CPU only.

## train the flow model

```
python train_flow_model.py
```

