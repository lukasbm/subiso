"""One module per engine, plus the router in :mod:`subiso.portfolio`.

``vf2`` (NetworkX)     attributes, capped/lazy enumeration, every graph
``vf3`` (``vf3py``)    existence and mappings; unlabelled, edgy graphs only (optional)
``lad`` (igraph)       existence, counting, mappings, anchors; materialises embeddings
``fastiso``            C++ induced-embedding counting; induced only (optional)
``smarts`` (RDKit)     SMARTS substructure matching over molecules
"""
