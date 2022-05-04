import datetime
import time
from collections import defaultdict
from enum import Enum
from itertools import chain, islice
from typing import Iterable, Optional, Union

import rubrix as rb
from rubrix.client.models import Record
from rubrix.client.sdk.commons.errors import NotFoundApiError


class Priority(Enum):

    Critical = 1
    High = 5
    Medium = 10
    Low = 30


def __batch_iterable__(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([next(batchiter)], batchiter)


class DatasetQueryStream:
    def __init__(
        self,
        dataset: str,
        query: str,
        priority: Priority = Priority.Medium,
        unique: bool = False,
        es_refresh_interval: int = 1,
        **query_params,
    ):
        self.dataset = dataset
        self.query = query
        self.query_params = query_params or {}
        self.priority = priority
        self.unique = unique
        self.wait_for_refresh_interval = es_refresh_interval
        self.__ids__ = defaultdict(lambda: False)

    def __check_query__(
        self,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ) -> Iterable[Record]:
        start_from = start.isoformat() if start else "*"
        end_to = end.isoformat() if end else "*"

        last_updated_query_part = f"[{start_from} TO {end_to}]"
        query = (
            self.query.format(**self.query_params) if self.query_params else self.query
        )
        query = f"{query} AND last_updated:{last_updated_query_part}"
        try:
            time.sleep(self.wait_for_refresh_interval)
            records = rb.load(
                self.dataset,
                query=query,
                as_pandas=False,
            )
            for record in records:
                if not self.unique or (self.unique and not self.__ids__[record.id]):
                    self.__ids__[record.id] = True
                    yield record

        except NotFoundApiError as ex:
            print(f"No dataset found. {ex}")

    def __call__(
        self,
        start_from: Optional[datetime.datetime] = None,
        batch_size: Optional[int] = None,
    ) -> Union[Iterable[Iterable[Record]], Iterable[Record]]:
        def inner_call():
            start = start_from
            end = datetime.datetime.utcnow()
            time2sleep = self.priority.value
            while True:
                results = self.__check_query__(start=start, end=end)
                start = end
                yield from results
                time.sleep(time2sleep)
                end = datetime.datetime.utcnow()

        if batch_size:
            return __batch_iterable__(inner_call(), size=batch_size)
        else:
            return inner_call()
