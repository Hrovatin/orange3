import logging
import hashlib
import pickle
from os.path import join, isfile

from Orange.canvas.config import cache_dir


log = logging.getLogger(__name__)


class EmbeddingCancelledException(Exception):
    """
    Thrown when the embedding task is cancelled from another thread.
    (i.e. ImageEmbedder.cancelled attribute is set to True).
    """


class EmbeddingConnectionError(ConnectionError):
    """
    Common error when embedding is interrupted because of connection problems
    or server unavailability - embedder do not respond.
    """


class EmbedderCache:

    _cache_file_blueprint = '{:s}_embeddings.pickle'

    def __init__(self, model):
        # init the cache

        cache_file_path = self._cache_file_blueprint.format(model)
        self._cache_file_path = join(cache_dir(), cache_file_path)
        self._cache_dict = self._init_cache()

    def _init_cache(self):
        if isfile(self._cache_file_path):
            try:
                return self.load_pickle(self._cache_file_path)
            except EOFError:
                return {}
        return {}

    @staticmethod
    def save_pickle(obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def md5_hash(bytes_):
        md5 = hashlib.md5()
        md5.update(bytes_)
        return md5.hexdigest()

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        self.save_pickle(self._cache_dict, self._cache_file_path)

    def get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

    def add(self, cache_key, value):
        self._cache_dict[cache_key] = value
