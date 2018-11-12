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

topics = [['creator', 'eat', 'asked', 'maybe', 'name', 'acting', 'rest', 'eyes', 'romans', 'control'],
['correct', 'say', 'son', 'unfortunate', 'perhaps', 'meat', 'gods', 'sin', 'considerations', 'anyway'],
['god', 'choices', 'refuse', 'maybe', 'mind', 'death', 'forever', 'friends', 'people', 'grave'],
['cool', 'nobody', 'whereas', 'teaches', 'doubts', 'also', 'forever', 'worry', 'romans', 'assume'],
['sea', 'keep', 'specific', 'believe', 'another', 'master', 'fall', 'caused', 'day', 'judge'],
['fall', 'grave', 'refuse', 'call', 'much', 'peace', 'going', 'though', 'anyway', 'day'],
['whatever', 'doubts', 'work', 'hand', 'directly', 'concerned', 'paul', 'least', 'jesus', 'sabbath'],
['eat', 'much', 'known', 'race', 'open', 'knew', 'serve', 'ends', 'choose', 'opinion'],
['chance', 'human', 'afraid', 'detail', 'music', 'hand', 'ever', 'son', 'considerations', 'content'],
['human', 'note', 'rest', 'person', 'groups', 'files', 'forever', 'short', 'day', 'others'],
['guilty', 'ca', 'sea', 'none', 'light', 'else', 'christians', 'many', 'open', 'ever'],
['master', 'let', 'believed', 'since', 'words', 'may', 'worry', 'work', 'hate', 'lord'],
['chance', 'drink', 'nobody', 'though', 'near', 'hear', 'considerations', 'kingdom', 'requirement', 'black'],
['eyes', 'eat', 'talking', 'master', 'therefore', 'may', 'holy', 'death', 'sure', 'words'],
['requirement', 'directly', 'enough', 'light', 'keeps', 'christ', 'clean', 'mind', 'justice', 'kingdom']]


# topics = [
# ["recently", "used", "dear", "sun", "having", "subject", "comp", "simple", "says", "circuit"],
# ["question", "use", "problem", "just", "version", "window", "info", "source"],
# ["like", "new", "article", "com", "christians", "discussion", "dec", "did", "argument", "hp"],
# ["trying", "think", "window", "yes", "god", "posted", "actually", "write", "wrong", "code"],
# ["need", "faq", "time", "apr", "archive", "ago", "93", "answer", "probably", "help"],
# ["looking", "want", "people", "point", "post", "lot", "true", "send", "build", "suggest"],
# ["ve", "try", "got", "seen", "posting", "motif", "reply", "wi", "look", "heard"],
# ["don", "know", "read", "sorry", "xt", "really", "say", "yo", "set", "recent"],
# ["just", "let", "tell", "sure", "right", "come", "idea", "running", "ok", "asked"],
# ["does", "know", "hello", "good", "deleted", "anybody", "folks", "way", "wondering", "stuff"]
# ]


print(topic_coherence(topics, services=["cv"]))
