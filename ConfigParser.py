import yaml
import argparse

from Paco_classifier.data_loader import FileSelectionMode, SampleExtractionMode

kPATH_IMAGES_DEFAULT = "datasets/images"
kPATH_REGION_MASKS_DEFAULT = "datasets/regions"
kPATH_BACKGROUND_DEFAULT = "datasets/layers/background"
kPATH_LAYERS_DEFAULT = ["datasets/layers/staff", "datasets/layers/neumes"]
kPATH_OUTPUT_MODELS_DEFAULT = ["Models/model_background.h5", "Models/model_staff.h5", "Models/model_neumes.h5"]
kBATCH_SIZE_DEFAULT = 8
kPATCH_HEIGHT_DEFAULT = 256
kPATCH_WIDTH_DEFAULT = 256
kMAX_NUMBER_OF_EPOCHS_DEFAULT = 1
kNUMBER_SAMPLES_PER_CLASS_DEFAULT = 100
kEARLY_STOPPING_PATIENCE_DEFAULT = 15
kFILE_SELECTION_MODE_DEFAULT = FileSelectionMode.SHUFFLE
kSAMPLE_EXTRACTION_MODE_DEFAULT = SampleExtractionMode.RANDOM


def getDefaultConfig():
    """Return default configuration.

    When using yaml, default config is just a dictionary.
    """
    tmp = {   'batch_size': kBATCH_SIZE_DEFAULT,
    'max_epochs': kBATCH_SIZE_DEFAULT,
    'number_samples_per_class': kNUMBER_SAMPLES_PER_CLASS_DEFAULT,
    'patch_height': kPATCH_HEIGHT_DEFAULT,
    'patch_width': kPATCH_WIDTH_DEFAULT,
    'path_bg': kPATH_BACKGROUND_DEFAULT,
    'path_layer': kPATH_LAYERS_DEFAULT,
    'path_out': kPATH_OUTPUT_MODELS_DEFAULT,
    'path_regions': kPATH_REGION_MASKS_DEFAULT,
    'path_src': kPATH_IMAGES_DEFAULT,
    'patience': kEARLY_STOPPING_PATIENCE_DEFAULT,
    'sample_extraction_mode': 'RANDOM',
    'file_selection_mode': 'SHUFFLE',
    #'path_ckpt': [],
    #'testset': [],
    }
    return tmp

def loadConfig(config_path, verbose=False):
    """Read config from yaml file.
    
    Read the yaml file and merge it with the default yaml. Return a argparse.Namespace.
    YAML is a good stuff, please use yaml.

    Parameters:
        config_path (str): Input yaml path.
    Returns:
        argparse.Namespace : This is a namespace. (?)
    """

    # Load default config
    config = getDefaultConfig()

    # Read from yaml file
    with open(config_path, "r") as fp:
        user_config = yaml.safe_load(fp)
    config.update(user_config)

    # Remove these weird enum classes!
    config = argparse.Namespace(**config)
    config.sample_extraction_mode = SampleExtractionMode.from_string(config.sample_extraction_mode)
    config.file_selection_mode = FileSelectionMode.from_string(config.file_selection_mode)

    if verbose:
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint ("= "*7)
        pp.pprint ("Load Config from {}".format(config_path))
        pp.pprint(config.__dict__)
        pp.pprint ("= "*7)

    return config