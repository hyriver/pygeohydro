class NLCD:
    """ A helper for proccessing NLCD data.
    References:
        https://github.com/jzmiller1/nlcd
        https://geopython.github.io/OWSLib/
        https://www.mrlc.gov/data-services-page
        https://www.arcgis.com/home/item.html?id=624863a9c2484741a9e2cc1ec9c95bce
        https://github.com/ozak/georasters
        https://automating-gis-processes.github.io/CSC18/index.html
        https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
    """

    def __init__(self):
        self.years = [2016, 2013, 2011, 2008, 2006, 2004, 2001]

        self.values = {
            "0": "Unclassified",
            "11": "Open Water",
            "12": "Perennial Ice/Snow",
            "21": "Developed, Open Space",
            "22": "Developed, Low Intensity",
            "23": "Developed, Medium Intensity",
            "24": "Developed High Intensity",
            "31": "Barren Land (Rock/Sand/Clay)",
            "41": "Deciduous Forest",
            "42": "Evergreen Forest",
            "43": "Mixed Forest",
            "51": "Dwarf Scrub",
            "52": "Shrub/Scrub",
            "71": "Grassland/Herbaceous",
            "72": "Sedge/Herbaceous",
            "73": "Lichens",
            "74": "Moss",
            "81": "Pasture/Hay",
            "82": "Cultivated Crops",
            "90": "Woody Wetlands",
            "95": "Emergent Herbaceous Wetlands",
        }

        self.categories = {
            "Unclassified": ("0"),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43"),
            "Shrubland": ("51", "52"),
            "Herbaceous": ("71", "72", "73", "74"),
            "Planted/Cultivated": ("81", "82"),
            "Wetlands": ("90", "95"),
        }

        self.roughness = {
            "11": 0.0250,
            "12": 0.0220,
            "21": 0.0400,
            "22": 0.1000,
            "23": 0.0800,
            "24": 0.1500,
            "31": 0.0275,
            "41": 0.1600,
            "42": 0.1800,
            "43": 0.1700,
            "52": 0.1000,
            "71": 0.0350,
            "81": 0.0325,
            "82": 0.0375,
            "90": 0.1200,
            "95": 0.0700,
        }
