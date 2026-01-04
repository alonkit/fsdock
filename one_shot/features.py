from datasets.process_chem.features import lig_feature_dims

lig_feature_dims[0].append(2)  # number of scalar features + one shot indicator


get_aux_lig_feature_dims = lambda: lig_feature_dims