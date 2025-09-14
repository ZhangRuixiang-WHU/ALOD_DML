# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class Dotav2Dataset(CocoDataset):

    # CLASSES = ('ship', 'bridge', 'oiltank', 'plane')
    CLASSES = ('large-vehicle', 'small-vehicle', 'plane', 'bridge', 'ship', 'harbor', 'swimming-pool', 'roundabout', 'soccer-ball-field', 
               'helicopter', 'storage-tank', 'tennis-court', 'baseball-diamond', 'basketball-court', 'ground-track-field','container-crane',
               'airport','helipad')

