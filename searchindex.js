Search.setIndex({"docnames": ["api_reference/assemble", "api_reference/dataset", "api_reference/functional", "api_reference/mesh", "api_reference/ode", "api_reference/operator", "api_reference/sparse", "get_started/benchmark", "get_started/introduction", "index", "install/installation", "tutorial/adjacency", "tutorial/linear_elasticity", "tutorial/poisson", "tutorial/wave"], "filenames": ["api_reference/assemble.rst", "api_reference/dataset.rst", "api_reference/functional.rst", "api_reference/mesh.rst", "api_reference/ode.rst", "api_reference/operator.rst", "api_reference/sparse.rst", "get_started/benchmark.rst", "get_started/introduction.rst", "index.rst", "install/installation.rst", "tutorial/adjacency.rst", "tutorial/linear_elasticity.rst", "tutorial/poisson.rst", "tutorial/wave.rst"], "titles": ["torch_fem.assemble", "torch_fem.dataset", "torch_fem.functional", "torch_fem.mesh", "torch_fem.ode", "torch_fem.operator", "torch_fem.sparse", "Benchmark", "Introduction by Example", "Torch-FEM Document", "Installation", "Adjacency", "Linear Elasticity", "Poisson Equation", "Wave Equation"], "terms": {"class": [0, 1, 3, 4, 5, 6, 12], "elementassembl": [0, 12], "quadrature_weight": 0, "quadrature_point": 0, "shape_v": 0, "projector": 0, "edg": [0, 6], "n_point": [0, 3], "sourc": [0, 1, 2, 3, 4, 5, 6], "base": [0, 1, 3, 4, 5, 6], "modul": [0, 3, 6], "The": [0, 1, 3, 6], "i": [0, 1, 2, 3, 4, 6, 10, 13, 14], "inherit": 0, "from": [0, 1, 4, 6, 12, 13, 14], "torch": [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14], "nn": [0, 3, 13], "therefor": 0, "all": [0, 1, 6], "oper": [0, 6, 9], "applic": 0, "you": [0, 3], "ar": [0, 3, 6], "encourag": 0, "build": 0, "directli": 0, "instead": [0, 3], "should": [0, 3, 6], "us": [0, 1, 3, 6], "from_mesh": [0, 12, 13, 14], "from_assembl": [0, 14], "mesh": [0, 9, 11, 12, 14], "output": 0, "when": [0, 3], "call": 0, "spars": [0, 4, 9], "matrix": [0, 2, 3, 4, 6, 12], "which": 0, "global": 0, "galerkin": 0, "shape": [0, 1, 2, 3, 4, 5, 6], "mathbb": [0, 4, 6], "r_": 0, "text": [0, 2, 4, 5, 6], "mathcal": [0, 1, 3, 6], "v": [0, 1, 2, 3], "im": 0, "h": [0, 6], "where": [0, 1, 2, 3, 4, 6], "number": [0, 1, 2, 3, 6], "degre": [0, 6], "freedom": 0, "per": 0, "point": [0, 1, 3, 11, 12, 13, 14], "k": [0, 1, 2, 4, 12, 13, 14], "overset": 0, "bsr": 0, "leftarrow": 0, "hat": 0, "k_": [0, 4, 5, 12, 14], "nkl": 0, "p_": 0, "e": [0, 1, 3, 6, 12], "nhij": 0, "local": 0, "hklij": 0, "ij": [0, 1, 2, 4, 6], "int_": 0, "omega": 0, "f": [0, 1, 4, 12, 13, 14], "u": [0, 1, 4, 5, 12, 13, 14], "dv": 0, "forward": [0, 4, 12], "function": [0, 1, 3, 4, 6, 9], "defin": 0, "thi": [0, 1, 3], "non": [0, 6], "zero": [0, 6, 12], "valu": [0, 1, 2, 3, 4, 5, 6, 11], "r": [0, 1, 3, 4], "time": [0, 1, 3, 4], "d": [0, 1, 2, 3, 4, 12], "each": [0, 1, 3, 6], "c": [0, 1, 2, 3, 4, 6, 12, 14], "project": 0, "tensor": [0, 1, 2, 3, 4, 5, 6, 12], "cell": [0, 3], "basi": [0, 2, 3], "connect": [0, 3], "vertic": [0, 1, 3], "mass": 0, "m_": [0, 4, 14], "u_i": 0, "v_j": 0, "import": [0, 1, 4, 11, 12, 13, 14], "massassembl": 0, "def": [0, 4, 12], "self": [0, 3, 4, 12], "return": [0, 1, 2, 3, 4, 5, 6, 12], "dot": [0, 2, 12, 14], "gen_rectangl": [0, 3, 13, 14], "m": [0, 1, 4, 6, 14], "laplac": 0, "nabla": 0, "cdot": [0, 1, 2, 4], "laplaceassembl": 0, "gradu": [0, 12], "gradv": [0, 12], "gen_circl": [0, 3], "type": [0, 1, 2, 3, 4, 5, 6, 12], "kei": [0, 3], "one": [0, 3, 4], "element_typ": [0, 1, 3, 11, 12], "correspond": [0, 3], "1d": [0, 1, 3, 4, 5, 6], "q": 0, "quadratur": 0, "weight": 0, "g": [0, 1, 3, 6], "triangle6": 0, "0": [0, 1, 2, 3, 4, 6, 11, 12, 13, 14], "5": [0, 1, 3, 11, 12], "bufferdict": [0, 3], "str": [0, 1, 3, 6], "2d": [0, 1, 3, 4], "dimens": [0, 1, 2, 3, 4], "domain": [0, 1], "b": [0, 2, 3, 4, 6, 12], "ach": 0, "object": [0, 1, 3, 4, 5, 6], "could": 0, "consid": [0, 3], "p_e": 0, "_": [0, 2, 6, 14], "c_e": 0, "b_e": 0, "rightarrow": 0, "set": [0, 3, 6], "1": [0, 1, 2, 3, 4, 6, 11, 12, 13], "2": [0, 1, 2, 3, 4, 6, 11, 12, 14], "3": [0, 1, 6, 11, 12, 13], "int": [0, 1, 2, 3, 4, 6], "either": 0, "quad9": 0, "list": [0, 2, 3, 6], "properti": [0, 3, 6], "devic": [0, 3, 6, 12], "cpu": [0, 3, 6], "cuda": [0, 3], "x": [0, 1, 2, 3, 6], "dtype": [0, 3, 6, 12], "data": [0, 3, 6], "float32": [0, 3, 6], "float64": [0, 3], "cast": 0, "paramet": [0, 1, 2, 3, 4, 5, 6, 12, 13], "buffer": [0, 3], "dst_type": 0, "method": [0, 3], "modifi": 0, "place": 0, "arg": 0, "string": 0, "desir": 0, "abstract": 0, "kwarg": 0, "weak": 0, "form": 0, "overrid": 0, "similar": 0, "can": [0, 1, 6, 10], "meth": 0, "__call__": 0, "option": [0, 1, 3, 6], "gradx": 0, "3d": [0, 1, 6], "point_data": [0, 3, 12], "dict": [0, 3], "pass": [0, 3, 4], "example_kei": 0, "gradexample_kei": 0, "4d": 0, "classmethod": [0, 3], "obj": [0, 3, 6], "an": 0, "anoth": [0, 6], "It": 0, "": [0, 4], "much": 0, "faster": 0, "than": 0, "alreadi": 0, "have": [0, 6], "sharig": 0, "same": [0, 3, 4, 6], "new": 0, "share": [0, 3, 6], "quadrature_ord": [0, 12, 14], "none": [0, 1, 3, 5, 6, 11], "slower": 0, "becaus": 0, "precomput": [0, 4, 12], "order": [0, 1, 3, 11], "poisit": 0, "integ": 0, "determin": [0, 1], "get_quadratur": 0, "topologi": 0, "nodeassembl": 0, "vector": [0, 2, 4], "n": [0, 1, 2, 4, 6, 14], "node_assembl": 0, "laplaceelementassembl": [0, 13, 14], "mathrm": 0, "masselementassembl": [0, 14], "meshgen": [1, 11], "chara_length": [1, 3, 11, 12, 13, 14], "cache_path": [1, 3], "tmp": [1, 3], "msh": [1, 3], "If": [1, 6], "gener": [1, 6], "mix": [1, 3], "otherwis": [1, 3, 6], "element": [1, 3, 6], "float": [1, 3, 4, 6], "characterist": [1, 3], "length": [1, 3], "smaller": 1, "more": 1, "dens": [1, 6], "default": [1, 3, 4, 6], "exampl": [1, 4, 6, 9], "rectangl": [1, 3], "triangl": 1, "tri": [1, 3, 11], "addrectangl": 1, "add": [1, 3], "gen": [1, 11], "plot": [1, 3, 12, 13, 14], "visual": [1, 3, 13], "left": [1, 3, 4], "right": [1, 3, 4, 5], "mesh_gen": [1, 11], "add_rectangl": [1, 11], "quad": [1, 3, 4, 11, 12], "remove_circl": [1, 11], "bottom": [1, 3], "width": 1, "height": 1, "geometri": 1, "boundari": [1, 3, 5], "itself": 1, "remove_rectangl": 1, "remov": 1, "add_circl": 1, "cx": [1, 3], "cy": [1, 3], "circl": [1, 3], "coordin": [1, 3], "center": [1, 3, 11], "y": [1, 3], "radiu": [1, 3], "cirlc": 1, "add_cub": 1, "z": [1, 3], "dx": 1, "dy": 1, "dz": 1, "cube": [1, 3], "onli": [1, 3, 5], "work": 1, "depth": 1, "remove_cub": 1, "add_spher": 1, "sphere": [1, 3], "remove_spher": 1, "show": [1, 3, 11], "fals": [1, 3, 6], "bool": [1, 3, 6, 12], "whether": [1, 3, 6], "gmsh": 1, "gui": 1, "poissonmultifrequ": [1, 13], "multi": 1, "frequenc": 1, "wave": [1, 9], "condit": [1, 5], "delta": 1, "x_1": 1, "x_2": 1, "t": [1, 4, 6], "pm": 1, "sampl": 1, "coeffici": 1, "randomli": [1, 6], "mu": 1, "sim": 1, "unif": 1, "ignor": 1, "random": [1, 6, 13, 14], "poisson": [1, 7, 9], "speed": 1, "initial_condit": [1, 13, 14], "frac": [1, 2, 4, 6], "pi": 1, "sum_": [1, 2, 4, 6], "j": [1, 2, 4, 6], "a_": [1, 2, 4, 6], "sin": 1, "ix": 1, "jy": 1, "must": [1, 3], "u0": [1, 4, 14], "solut": [1, 4, 5, 6, 13, 14], "tenor": 1, "heatmultifrequ": 1, "heat": 1, "partial": [1, 4], "mu_m": 1, "sqrt": [1, 6], "2m": 1, "2t": 1, "mx_2": 1, "ut": 1, "wavemultifrequ": 1, "u_": [1, 5, 12, 14], "tt": 1, "initi": [1, 4], "v0": [1, 14], "co": 1, "trace": 2, "A": [2, 4, 6, 14], "ii": 2, "reduce_dim": [2, 12], "ab": 2, "ai": 2, "b_": [2, 4], "bi": 2, "ddot": [2, 4], "aij": 2, "bij": 2, "mul": [2, 14], "n_basi": [2, 12], "ey": [2, 6], "dim": [2, 6], "begin": [2, 4], "case": 2, "v_": 2, "neq": [2, 6], "end": [2, 4], "fill": [2, 6], "sym": 2, "bmatrix": [2, 4], "vdot": [2, 4], "n_": [2, 5, 6], "row": [2, 6], "col": [2, 6], "transpos": [2, 6], "ji": 2, "matmul": [2, 12], "ik": 2, "kj": 2, "meshio": 3, "space": 3, "cell_data": 3, "field_data": 3, "field": 3, "cell_set": 3, "dim2eletyp": 3, "default_eletyp": 3, "default_element_typ": 3, "register_point_data": 3, "pair": 3, "sinc": 3, "recommend": 3, "__setitem__": 3, "node": 3, "regist": 3, "to_meshio": 3, "save": 3, "file_nam": 3, "file_format": 3, "name": 3, "file": 3, "format": [3, 6], "vtk": 3, "extens": 3, "to_fil": 3, "node_adjac": [3, 11], "get": 3, "adjac": [3, 9], "insid": 3, "fulli": 3, "iter": 3, "sparsematrix": [3, 4, 6], "element_adjac": [3, 11], "thei": 3, "facet": 3, "map": [3, 4], "do": [3, 4], "abov": 3, "save_path": [3, 12, 14], "backend": [3, 6, 12, 14], "matplotlib": [3, 11, 12, 13, 14], "dt": [3, 4, 14], "show_mesh": [3, 12, 14], "static": [3, 5, 6], "subplot": [3, 13], "mp4": [3, 13, 14], "gif": 3, "item": [3, 13], "path": [3, 14], "endswith": 3, "pyvista": 3, "interv": 3, "between": 3, "frame": 3, "boundary_mask": [3, 13, 14], "mask": [3, 5, 6], "is_boundari": 3, "requir": [3, 6], "compos": 3, "from_meshio": 3, "read": 3, "from_fil": 3, "top": [3, 4, 6], "decid": 3, "dataset": [3, 9, 13], "gen_hollow_rectangl": [3, 12], "outer_left": 3, "outer_right": 3, "outer_bottom": 3, "outer_top": 3, "inner_left": 3, "25": 3, "inner_right": 3, "75": 3, "inner_bottom": 3, "inner_top": 3, "outer": 3, "inner": [3, 5], "gen_hollow_circl": 3, "r_inner": 3, "r_outer": 3, "gen_l": 3, "top_inn": 3, "right_inn": 3, "gen_cub": 3, "front": 3, "back": 3, "gen_hollow_cub": 3, "outer_front": 3, "outer_back": 3, "inner_front": 3, "inner_back": 3, "gmsh_cach": 3, "gen_spher": 3, "cz": 3, "gen_hollow_spher": 3, "arrai": 4, "textbf": 4, "mathfrak": 4, "hline": 4, "ccc": 4, "c_1": 4, "11": 4, "c_": 4, "s1": 4, "ss": 4, "b_1": 4, "qquad": 4, "k_i": 4, "c_i": 4, "tau": 4, "k_j": 4, "psi": 4, "b_i": 4, "sum_j": 4, "explicitrungekutta": 4, "shoul": 4, "lower": 4, "triangular": 4, "21": 4, "forward_m": 4, "side": [4, 5], "normal": 4, "problem": [4, 7], "assum": 4, "ident": 6, "diagon": [4, 6], "step": [4, 13], "t0": 4, "explicit": 4, "t_0": 4, "implicitlinearrungekutta": 4, "forward_a": 4, "comput": 4, "linear": [4, 5, 9], "term": 4, "forward_b": 4, "expliciteul": 4, "approx": 4, "myexpliciteul": 4, "rand": 4, "4": [4, 6, 14], "ut_gt": 4, "ut_mi": 4, "assert": 4, "allclos": 4, "implicitlineareul": 4, "w": 4, "myimplicitlineareul": 4, "doubl": [4, 12], "expect": 4, "got": 4, "midpointlineareul": 4, "mymidpointlineareul": 4, "dirichlet_mask": [5, 12], "dirichlet_valu": 5, "dirichlet": 5, "f_": [5, 12, 14], "ou2in": 5, "dof": 5, "outer_dof": 5, "condense_rh": [5, 14], "rh": 5, "hand": [4, 5], "system": [4, 5], "inner_dof": 5, "recov": [5, 12, 14], "recovert": 5, "edata": 6, "coo": 6, "indic": 6, "column": 6, "tupl": 6, "first": 6, "two": 6, "hash_layout": 6, "check": 6, "matric": 6, "layout": 6, "hash": 6, "elementwise_oper": 6, "func": 6, "elementwis": 6, "scalar": 6, "callabl": 6, "result": 6, "solv": [4, 6, 7, 12, 13, 14], "requires_grad_": 6, "requires_grad": 6, "true": [6, 12, 13, 14], "gradient": [3, 6], "tranpos": 6, "wise": 6, "squar": 6, "root": 6, "reciproc": 6, "axi": [6, 11], "how": 6, "mani": 6, "sum": 6, "grad": 6, "grad_fn": 6, "autograd": 6, "nnz": 6, "layout_mask": 6, "detach": 6, "to_scipy_coo": 6, "scipi": 6, "coo_matrix": 6, "to_sparse_coo": 6, "turn": 6, "sparse_coo_tensor": 6, "lost": 6, "rtype": [3, 6], "to_dens": 6, "maintain": 6, "has_same_layout": 6, "compar": 6, "layout_hash": 6, "from_sparse_coo": 6, "convert": 6, "from_block_coo": 6, "block": 6, "size": 6, "from_dens": 6, "densiti": 6, "param": 6, "random_layout": 6, "random_from_layout": 6, "full": 6, "combine_vector": 6, "combin": 6, "combine_matrix": 6, "dispatch": 6, "attribut": [], "introduct": 9, "benchmark": 9, "equat": 9, "elast": 9, "torch_fem": [9, 10, 11, 12, 13, 14], "assembl": [9, 13], "od": 9, "avail": 10, "pip": 10, "pyplot": [11, 13], "plt": [11, 13], "adj": 11, "draw_graph": 11, "numpi": [11, 13], "draw_mesh": [11, 13], "cat": 11, "mean": 11, "n_element": 11, "kassembl": 12, "nu": 12, "n_dim": 12, "zeros_lik": [12, 14], "02": 12, "k_asm": 12, "is_outer_top_boundari": 12, "05": 12, "flatten": 12, "is_outer_bottom_boundari": 12, "condens": [4, 12, 13, 14], "ux": 12, "ui": 12, "press_hollow_rectangl": 12, "png": [12, 13], "np": 13, "tqdm": 13, "streamplott": 13, "__name__": [13, 14], "__main__": [13, 14], "manual_se": [13, 14], "123456": [13, 14], "8": 13, "optim": 13, "adam": 13, "lr": 13, "1e": 13, "schedul": 13, "lr_schedul": 13, "steplr": 13, "step_siz": 13, "10": 13, "gamma": 13, "9": 13, "epoch": 13, "100": [13, 14], "loss_fn": 13, "mseloss": 13, "loss": 13, "filenam": 13, "plotter": 13, "pbar": 13, "total": 13, "rang": [13, 14], "zero_grad": 13, "todo": 13, "why": 13, "retain_graph": 13, "backward": 13, "set_postfix": 13, "updat": 13, "append": [13, 14], "fig": 13, "ax": 13, "figsiz": 13, "12": 13, "arang": 13, "len": 13, "label": 13, "set_xlabel": 13, "set_ylabel": 13, "legend": 13, "set_yscal": 13, "log": 13, "savefig": 13, "your": [13, 14], "browser": [13, 14], "doe": [13, 14], "support": [13, 14], "video": [13, 14], "tag": [13, 14], "sy": 14, "wavemultisinco": 14, "util": 14, "001": 14, "01": 14, "m_asm": 14, "a_asm": 14, "u1": 14, "u2": 14, "us_gt": 14, "predict": 14, "ground": 14, "truth": 14, "clone": [3, 6], "vanish": 3, "so": 3, "we": 3, "provid": 3, "min": [], "ntime": [], "ain": [], "bin": [], "uin": [], "m_0": 4, "a_0tau": [], "a_1tau": [], "m_1": 4, "k_0": 4, "k_1": 4, "b_0": 4, "a_0": 4, "a_1": 4, "pre_solve_lh": 4, "someth": 4, "befor": 4, "after": 4, "pre_solve_rh": 4, "post_solv": 4, "postprocess": 4, "recoveri": 4, "torch_scipi": 6, "origin": 6, "attr": 6}, "objects": {"torch_fem.assemble.builtin": [[0, 0, 1, "", "LaplaceElementAssembler"], [0, 0, 1, "", "MassElementAssembler"]], "torch_fem.assemble.builtin.LaplaceElementAssembler": [[0, 1, 1, "", "forward"]], "torch_fem.assemble.builtin.MassElementAssembler": [[0, 1, 1, "", "forward"]], "torch_fem.assemble.element_assembler": [[0, 0, 1, "", "ElementAssembler"]], "torch_fem.assemble.element_assembler.ElementAssembler": [[0, 2, 1, "", "device"], [0, 3, 1, "", "dimension"], [0, 2, 1, "", "dtype"], [0, 3, 1, "", "edges"], [0, 3, 1, "", "element_types"], [0, 3, 1, "", "elements"], [0, 1, 1, "", "forward"], [0, 1, 1, "", "from_assembler"], [0, 1, 1, "", "from_mesh"], [0, 3, 1, "", "n_points"], [0, 3, 1, "", "projector"], [0, 3, 1, "", "quadrature_points"], [0, 3, 1, "", "quadrature_weights"], [0, 3, 1, "", "shape_val"], [0, 1, 1, "", "type"]], "torch_fem.assemble.node_assembler": [[0, 0, 1, "", "NodeAssembler"]], "torch_fem.assemble.node_assembler.NodeAssembler": [[0, 3, 1, "", "dimension"], [0, 3, 1, "", "element_types"], [0, 3, 1, "", "elements"], [0, 1, 1, "", "forward"], [0, 1, 1, "", "from_assembler"], [0, 1, 1, "", "from_mesh"], [0, 3, 1, "", "n_points"], [0, 3, 1, "", "projector"], [0, 3, 1, "", "quadrature_points"], [0, 3, 1, "", "quadrature_weights"], [0, 3, 1, "", "shape_val"], [0, 1, 1, "", "type"]], "torch_fem.dataset.equation": [[1, 0, 1, "", "HeatMultiFrequency"], [1, 0, 1, "", "PoissonMultiFrequency"], [1, 0, 1, "", "WaveMultiFrequency"]], "torch_fem.dataset.equation.HeatMultiFrequency": [[1, 1, 1, "", "initial_condition"], [1, 1, 1, "", "solution"]], "torch_fem.dataset.equation.PoissonMultiFrequency": [[1, 1, 1, "", "initial_condition"], [1, 1, 1, "", "solution"]], "torch_fem.dataset.equation.WaveMultiFrequency": [[1, 1, 1, "", "initial_condition"], [1, 1, 1, "", "solution"]], "torch_fem.dataset.mesh": [[1, 0, 1, "", "MeshGen"]], "torch_fem.dataset.mesh.MeshGen": [[1, 1, 1, "", "add_circle"], [1, 1, 1, "", "add_cube"], [1, 1, 1, "", "add_rectangle"], [1, 1, 1, "", "add_sphere"], [1, 1, 1, "", "gen"], [1, 1, 1, "", "remove_circle"], [1, 1, 1, "", "remove_cube"], [1, 1, 1, "", "remove_rectangle"], [1, 1, 1, "", "remove_sphere"]], "torch_fem.functional": [[2, 4, 0, "-", "assemble_helpers"]], "torch_fem.functional.assemble_helpers": [[2, 5, 1, "", "ddot"], [2, 5, 1, "", "dot"], [2, 5, 1, "", "eye"], [2, 5, 1, "", "matmul"], [2, 5, 1, "", "matrix"], [2, 5, 1, "", "mul"], [2, 5, 1, "", "sym"], [2, 5, 1, "", "trace"], [2, 5, 1, "", "transpose"], [2, 5, 1, "", "vector"]], "torch_fem.mesh": [[3, 0, 1, "", "Mesh"]], "torch_fem.mesh.Mesh": [[3, 2, 1, "", "boundary_mask"], [3, 3, 1, "", "cell_data"], [3, 3, 1, "", "cell_sets"], [3, 3, 1, "", "cells"], [3, 1, 1, "", "clone"], [3, 2, 1, "id0", "default_element_type"], [3, 3, 1, "", "default_eletyp"], [3, 2, 1, "", "device"], [3, 3, 1, "", "dim2eletyp"], [3, 2, 1, "", "dtype"], [3, 1, 1, "", "element_adjacency"], [3, 1, 1, "", "elements"], [3, 3, 1, "", "field_data"], [3, 1, 1, "", "from_file"], [3, 1, 1, "", "from_meshio"], [3, 1, 1, "", "gen_L"], [3, 1, 1, "", "gen_circle"], [3, 1, 1, "", "gen_cube"], [3, 1, 1, "", "gen_hollow_circle"], [3, 1, 1, "", "gen_hollow_cube"], [3, 1, 1, "", "gen_hollow_rectangle"], [3, 1, 1, "", "gen_hollow_sphere"], [3, 1, 1, "", "gen_rectangle"], [3, 1, 1, "", "gen_sphere"], [3, 2, 1, "", "n_point"], [3, 1, 1, "", "node_adjacency"], [3, 1, 1, "", "plot"], [3, 3, 1, "", "point_data"], [3, 3, 1, "", "points"], [3, 1, 1, "", "read"], [3, 1, 1, "", "register_point_data"], [3, 1, 1, "", "save"], [3, 1, 1, "", "to_file"], [3, 1, 1, "", "to_meshio"]], "torch_fem.ode": [[4, 0, 1, "", "ExplicitRungeKutta"], [4, 0, 1, "", "ImplicitLinearRungeKutta"], [4, 4, 0, "-", "builtin"]], "torch_fem.ode.ExplicitRungeKutta": [[4, 1, 1, "", "forward"], [4, 1, 1, "", "step"]], "torch_fem.ode.ImplicitLinearRungeKutta": [[4, 1, 1, "", "forward_A"], [4, 1, 1, "", "forward_B"], [4, 1, 1, "", "forward_M"], [4, 1, 1, "", "post_solve"], [4, 1, 1, "", "pre_solve_lhs"], [4, 1, 1, "", "pre_solve_rhs"], [4, 1, 1, "", "step"]], "torch_fem.ode.builtin": [[4, 0, 1, "", "ExplicitEuler"], [4, 0, 1, "", "ImplicitLinearEuler"], [4, 0, 1, "", "MidPointLinearEuler"]], "torch_fem.operator": [[5, 0, 1, "", "Condenser"]], "torch_fem.operator.Condenser": [[5, 1, 1, "", "condense_rhs"], [5, 3, 1, "", "dirichlet_mask"], [5, 3, 1, "", "dirichlet_value"], [5, 1, 1, "", "recover"]], "torch_fem.sparse": [[6, 0, 1, "", "SparseMatrix"]], "torch_fem.sparse.SparseMatrix": [[6, 2, 1, "", "T"], [6, 1, 1, "", "clone"], [6, 3, 1, "", "col"], [6, 1, 1, "", "combine"], [6, 1, 1, "", "combine_matrix"], [6, 1, 1, "", "combine_vector"], [6, 1, 1, "", "degree"], [6, 1, 1, "", "detach"], [6, 2, 1, "", "device"], [6, 2, 1, "", "dtype"], [6, 3, 1, "", "edata"], [6, 2, 1, "", "edges"], [6, 1, 1, "", "elementwise_operation"], [6, 1, 1, "", "eye"], [6, 1, 1, "", "from_block_coo"], [6, 1, 1, "", "from_dense"], [6, 1, 1, "", "from_sparse_coo"], [6, 1, 1, "", "full"], [6, 2, 1, "", "grad"], [6, 2, 1, "", "grad_fn"], [6, 1, 1, "", "has_same_layout"], [6, 3, 1, "", "hash_layout"], [6, 2, 1, "", "layout_mask"], [6, 2, 1, "", "nnz"], [6, 1, 1, "", "random"], [6, 1, 1, "", "random_from_layout"], [6, 1, 1, "", "random_layout"], [6, 1, 1, "", "reciprocal"], [6, 2, 1, "", "requires_grad"], [6, 1, 1, "", "requires_grad_"], [6, 3, 1, "", "row"], [6, 3, 1, "", "shape"], [6, 1, 1, "", "solve"], [6, 1, 1, "", "sqrt"], [6, 1, 1, "", "sum"], [6, 1, 1, "", "to_dense"], [6, 1, 1, "", "to_scipy_coo"], [6, 1, 1, "", "to_sparse_coo"], [6, 1, 1, "", "transpose"], [6, 1, 1, "", "type"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:property", "3": "py:attribute", "4": "py:module", "5": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "property", "Python property"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "module", "Python module"], "5": ["py", "function", "Python function"]}, "titleterms": {"torch_fem": [0, 1, 2, 3, 4, 5, 6], "assembl": [0, 2, 7], "content": [0, 1, 2, 3, 4, 5, 6], "element": [0, 11], "node": [0, 11], "built": [0, 4], "dataset": [1, 14], "mesh": [1, 3, 8, 13], "equat": [1, 8, 13, 14], "function": 2, "helper": 2, "od": 4, "rung": 4, "kutta": 4, "method": 4, "oper": 5, "condens": 5, "spars": 6, "sparsematirx": 6, "benchmark": 7, "speed": 7, "introduct": 8, "exampl": 8, "gener": 8, "partial": 8, "differenti": 8, "visual": 8, "result": 8, "torch": 9, "fem": 9, "document": 9, "instal": [9, 10], "get": 9, "start": 9, "tutori": 9, "api": 9, "refer": 9, "installatoion": 10, "via": 10, "pypi": 10, "adjac": 11, "linear": 12, "elast": 12, "press": 12, "hollow": 12, "rectangl": 12, "poisson": 13, "adapt": 13, "refin": 13, "wave": 14, "multi": 14, "frequenc": 14, "pipelin": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx.ext.intersphinx": 1, "nbsphinx": 4, "sphinx": 56}})