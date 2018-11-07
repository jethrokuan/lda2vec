import requests
import numpy as np


def get_request(url):
    for _ in range(5):
        try:
            return float(requests.get(url).text)
        except:
            pass
    return None


def topic_coherence(lists, services=['ca', 'cp', 'cv', 'npmi', 'uci',
                                     'umass']):
    """ Requests the topic coherence from AKSW Palmetto
    Arguments
    ---------
    lists : list of lists
        A list of lists with one list of top words for each topic.
    >>> topic_words = [['cake', 'apple', 'banana', 'cherry', 'chocolate']]
    >>> topic_coherence(topic_words, services=['cv'])
    {(0, 'cv'): 0.5678879445677241}
    """
    url = "http://palmetto.aksw.org/palmetto-webapp/service/{}?words={}"
    res = []
    for topic in lists:
        for s in services:
            u = url.format(s, '%20'.join(topic[:10]))
            print(u)
            req = requests.get(u)
            res.append(req.json())
    print("Average: {}".format(np.average(res)))
    print("Median: {}".format(np.median(res)))
    return res

topics = [["edu", "space", "com", "information", "available", "mail", "data", "ftp", "pub", "send"],
["don", "just", "like", "know", "think", "good", "time", "people", "ve", "going"],
["god", "people", "does", "jesus", "believe", "think", "say", "don", "just", "know"],
["thanks", "use", "window", "windows", "does", "know", "help", "like", "program", "using"],
["g9v", "b8f", "a86", "145", "1d9", "pl", "0t", "cx", "34u", "2di"],
["key", "use", "chip", "encryption", "used", "keys", "clipper", "bit", "bike", "number"],
["10", "00", "15", "25", "12", "20", "11", "16", "14", "13"],
["game", "year", "team", "games", "play", "season", "hockey", "league", "players", "win"],
["people", "government", "mr", "gun", "law", "president", "armenian", "said", "state", "israel"],
["file", "drive", "card", "disk", "scsi", "dos", "mac", "pc", "memory", "use"]]

print(topic_coherence(topics, services=["cv"]))
