# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset, ROTATED_DATASETS, ROTATED_PIPELINES # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .dota_mixpoint import DOTADataset_P
from .dota15 import DOTAV15Dataset
from .dotav2 import DOTAV2Dataset
from .dior import DIORDataset
from .soda import SODADataset
from .fair1m import FAIR1MDataset
from .aitodr import AITODRDataset
from .cocor import COCORDataset
from .dtod import DTODDataset
from .skur import SKURDataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403

# __all__ = ['SARDataset', 'DOTADataset', 'DOTAV2Dataset','SODADataset','build_dataset', 'HRSCDataset','ROTATED_DATASETS','ROTATED_PIPELINES']
__all__ = ['SARDataset', 'DOTADataset', 'DOTAV15Dataset', 'DOTAV2Dataset','SODADataset','AITODRDataset','COCORDataset','DTODDataset', 'DOTADataset_P',
           'DIORDataset', 'FAIR1MDataset', 'SKURDataset', 'build_dataset', 'HRSCDataset','ROTATED_DATASETS','ROTATED_PIPELINES']
