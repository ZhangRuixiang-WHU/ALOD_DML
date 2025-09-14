from .dota import DOTADataset
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class FAIR1MDataset(DOTADataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES =  ('LiquidCargoShip', 'PassengerShip', 'DryCargoShip', 'CargoTruck', 'SmallCar', 
                'DumpTruck', 'Van', 'Motorboat', 'Bridge', 'Intersection', 'Excavator', 'other-vehicle', 
                'Boeing737', 'A321', 'TennisCourt', 'BasketballCourt', 'Bus', 'A220', 'other-ship', 
                'other-airplane', 'Boeing787', 'FootballField', 'EngineeringShip', 'Warship', 'Tugboat', 
                'Tractor', 'Roundabout', 'ARJ21', 'FishingBoat', 'Boeing747', 'Trailer', 'TruckTractor', 
                'BaseballField', 'A330', 'A350', 'Boeing777', 'C919')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (0, 184, 255), (0, 42, 255), (45, 58, 255),
               (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255), (0, 184, 255), (0, 42, 255), (45, 58, 255),(165, 42, 42)]
