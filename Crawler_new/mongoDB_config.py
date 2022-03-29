import pymongo
from pymongo import *
from Web_dict import list_dict_web


def mongodb_connection(name):
    client = pymongo.MongoClient(
        "mongodb+srv://hunguet:hung1234@crawlnews.cw7k5.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    data_base = client[name]
    return data_base


class MongoDB_config():

    def __init__(self, data_base_name, collections):
        self.data_base = mongodb_connection(data_base_name)
        self.list_dict = list_dict_web
        self.collections = self.data_base[collections]

    def update_web(self):
        for i in self.list_dict:
            self.collections.update_one(
                {"dommain": i["dommain"]},
                {
                    "$set": {"dommain": i["dommain"],
                             "title": i["title"],
                             "summary": i["summary"],
                             "conetent": i["conetent"]}
                },
                upsert=True
            )

    def get_collections_web(self):
        return self.collections


# if __name__ == "__main__":
#     MongoDB_config("Web_Crawl", "Web_Struct").update_web()
