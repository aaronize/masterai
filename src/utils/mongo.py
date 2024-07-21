from pymongo import MongoClient

from pymongo.collection import Collection
from pymongo.database import Database

from src.utils.env import MONGODB_CONFIG

username = MONGODB_CONFIG.get("username")
password = MONGODB_CONFIG.get("password")
host = MONGODB_CONFIG.get("host")
port = MONGODB_CONFIG.get("port")

client = MongoClient(f"mongodb://{username}:{password}@{host}:{port}/?")

DB: Database = client["pcap_data"]
Events: Collection = DB.get_collection("events")
# DATASET: Collection = DB.get_collection("dataset")


if __name__ == '__main__':
    # DATASET.insert_one({"author": "Mike", "tags": ["mongodb", "python", "pymongo"]})
    # data = data
    pass
