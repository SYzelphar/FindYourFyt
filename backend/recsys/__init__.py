"""FindYourFit v2 recommendation engine."""
import os

# numpy, FAISS, scikit-learn and PyTorch each bring their own BLAS/OpenMP thread pools. On the
# small matrices used per request (the 133x133 posterior solve) they fight each other: a solve
# took ~300 ms instead of <1 ms. Requests are small and concurrent, so single-threaded BLAS is
# the right default. The env var covers libraries loaded later; threadpool_limits covers the
# ones already loaded by the time this package is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from threadpoolctl import threadpool_limits  # noqa: E402

from .bundle import Bundle, BundleUnavailable  # noqa: E402
from .recommender import RecConfig, Recommender  # noqa: E402
from .session_store import VALID_DIRECTIONS, InMemorySessionStore, SessionState  # noqa: E402

threadpool_limits(limits=1, user_api="blas")

__all__ = ["Bundle", "BundleUnavailable", "RecConfig", "Recommender", "VALID_DIRECTIONS",
           "InMemorySessionStore", "SessionState"]
