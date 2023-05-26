import json
import re
from typing import Literal

import urllib.request
from clustering_pipelines.dataset_loaders import DatasetLoader
from clustering_pipelines.utils.dataset_utils import load_and_extract


class AGNewsLoader(DatasetLoader):
    """
    Loader for AG News dataset:
    # http://http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    
    """
    def __init__(
            self,
            base_name: str = "ag_news",
            split: Literal["test", "train"] = "test",
            remove: tuple[str] = tuple()  # title
        ):
        if not split in ["test", "train"]:
            raise ValueError(
                f"`split` must be 'test' or 'train', got {split} instead")
        self.split = split
        self.remove = remove
        super().__init__(name=f"{base_name}_{split}")

    @property
    def _dataset_name(self) -> str:
        return f"ag_news"
    
    @property
    def url(self) -> str:
        return "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms"
    
    def _load(self) -> None:
        load_and_extract(self.url, self.path, "tar")

    def _parse(self) -> tuple[list, list]:
        classes = []
        texts = []

        with open(self.path / "classes.txt", "r") as f:
            label_to_class = {
                label+1:class_name.split("\n")[0]
                for label, class_name in enumerate(f)
            }

        with open(self.path / f"{self.split}.txt", "r") as f:
            for line in f:
                line = re.sub(r"(\S+)\\(\S+)", r"\1 \2", line[1:-2])
                label, title, text = line.split('","')
                classes.append(label_to_class[int(label)])

                if not "title" in self.remove:
                    text = title + ". " + text

                texts.append(text)

        return texts, classes


class BBCNewsLoader(DatasetLoader):
    """Loader for BBC News dataset: http://mlg.ucd.ie/datasets/bbc.html."""
    def __init__(
        self,
        name: str = "bbc_news",
        remove: tuple[str] = tuple()  # title
    ):
        self.remove = remove
        super().__init__(name=name)

    @property
    def _dataset_name(self) -> str:
        return "bbc_news"

    @property
    def url(self) -> str:
        return "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"

    def _parse(self) -> tuple[list, list]:
        classes = []
        texts = []

        for p in self.path.rglob('**/*.txt'):
            classes.append(p.parent.name)
            with open(p, "r") as f:
                text = f.read().replace("\n\n", "\n")

                if "title" in self.remove:
                    text = text.split("\n", 1)[-1]
                
                texts.append(text)

        return texts, classes
    

class NewsgroupsLoader(DatasetLoader):
    """
    Loader for 20 Newsgroups dataset:
    http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz.
    
    """
    def __init__(
        self,
        name: str = "20_newsgroups",
        remove: tuple[str] = tuple(),  # ‘headers’, ‘footers’, ‘quotes’
    ):
        self.remove = remove
        super().__init__(name=name)

    @property
    def _dataset_name(self) -> str:
        return "20_newsgroups"

    @property
    def url(self) -> str:
        return "http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz"

    def _parse(self) -> tuple[list, list]:
        classes = []
        texts = []

        generator = filter(lambda p: p.is_file(), self.path.rglob("**/*"))
        for p in generator:
            classes.append(p.parent.name)
            with open(p, "r", encoding="cp1251") as f:
                text = f.read()
            
            if "quotes" in self.remove:
                text = re.sub(r"\n(In article|>)[^\n]*", "", text)

            if "footers" in self.remove:
                text = text.rsplit("\n\n", 1)[0]

            if "headers" in self.remove:
                text = text.split("\n\n", 1)[-1]

            texts.append(text)

        return texts, classes
    

class TwitterSummarization(DatasetLoader):
    """
    Loader for TwitterSummarization dataset:
    https://github.com/cocoxu/twittersummarization.git
     
    """

    _filenames = [
        "yoga_20130116.json",
        "miami_20130116.json",
        "rob_20130116.json",
        "justin_20130116.json",
        "ms_20130116.json",
        "chicago_20130116.json",
        "fox_20130116.json",
        "india_20130116.json",
        "malaysia_20130116.json",
        "ipad_20130116.json",
        "west_ham_20130116.json",
        "golden_disk_awards_20130116.json",
        "skype_20130116.json",
        "iphone_20130116.json",
        "arsenal_20130116.json",
        "florida_20130116.json",
        "washington_20130116.json",
        "tokyo_20130116.json",
        "fa_cup_20130116.json",
        "instagram_20130116.json",
        "jesus_20130116.json",
        "obama_20130116.json",
        "swansea_20130116.json",
        "birmingham_20130116.json",
        "english_20130116.json",
        "school_20130116.json",
        "google_20130116.json",
        "new_york_20130116.json",
        "chelsea_20130116.json",
        "espn_20130116.json",
        "les_mis_20130116.json",
        "@youtube_20130116.json",
        "american_idol_20130116.json",
        "pll_20130116.json",
        "gda_20130116.json",
        "workaholics_20130116.json",
        "japan_20130116.json",
        "vegas_20130116.json"
    ]
    _base_url = "https://raw.githubusercontent.com/cocoxu/twittersummarization/master/NAACL_2013_evaluation/original_tweets_20130116/"

    def __init__(
        self,
        name: str = "twitter_summarization",
        remove: tuple[str]  = tuple()  # 'retweets', 'links', 'tech_symbols'
    ):
        self.remove = remove
        super().__init__(name=name)

    @property
    def _dataset_name(self) -> str:
        return "twitter_summarization"

    @property
    def url(self) -> str:
        return "https://github.com/cocoxu/twittersummarization.git"

    def _load(self) -> str:
        for filename in self._filenames:
            urllib.request.urlretrieve(
                url=self._base_url + filename,
                filename=self.path / filename
            )
        
    def _parse(self) -> tuple[list, list]:
        classes = []
        texts = []

        for json_path in self.path.rglob('**/*.json'):
            with open(json_path, "r") as f:
                d = json.load(f)
            class_name = re.sub(r'[^a-z]', '', json_path.stem)
            for el in d[0]:
                for el_ in el['tweets']:
                    text: str = el_['text']

                    if "retweets" in self.remove and text.startswith("RT @"):
                        continue
                    
                    if "links" in self.remove:
                        text = re.sub(r":?\s?https?:\/\/\S*", "", text)

                    if "tech_symbols" in self.remove:
                        text = re.sub(r"[@#]", "", text)

                    texts.append(text)
                    classes.append(class_name)
        
        return texts, classes
