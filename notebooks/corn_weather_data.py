import sys
sys.path.insert(0, '..')
import src.data.weather_script as ws

# m = 200
# cropValue = 1
# BAP = ws.bigArrayParser("../data/2023_30m_cdls.tif", m, cropValue)


# import pickle
# with open("BAP.pkl", "wb") as f:
#     pickle.dump(BAP, f)
#
# import pickle
# with open("my_object.pkl", "rb") as f:
#     BAP = pickle.load(f)

# k = 10
# r = 7
# cadot = BAP.get_area_with_climate(k, r, m, ["IA", "IL", "IN", "MO", "SD", "KS", "MN"], "2014-01-01", "2024-12-31")
#
# import shelve
# big_dict = cadot
#
# with shelve.open('big_dict.db') as db:
#     for key, value in big_dict.items():
#         db[str(key)] = value  # convert key to string

# import shelve
# with shelve.open('big_dict.db') as db:
#     cadot = db
#     proj = ws.get_projections_multithreaded(cadot)
#     with shelve.open('proj_dict.db') as db:
#         for key, value in proj.items():
#             db[str(key)] = value  # convert key to string
import shelve
with shelve.open('proj_dict.db') as db:
    proj = db
    features = ws.get_weather_features_multithreaded(proj)
    with shelve.open('feat_dict.db') as db:
        for key, value in features.items():
            db[str(key)] = value  # convert key to string
