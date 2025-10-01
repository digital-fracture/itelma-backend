import asyncio
from collections.abc import Iterable
from typing import ClassVar, Self, override
from warnings import deprecated


class CustomAsyncioPriorityQueue[T](asyncio.PriorityQueue[tuple[int, T]]):
    """An ``asyncio.PriorityQueue`` that prioritizes items based on their type."""

    # user should override this in subclasses to define priority order
    # each entry may be a single `type` or an iterable of types
    type_priority_order: ClassVar[Iterable[type | Iterable[type]]] = ()

    _type_to_priority: ClassVar[dict[type, int]] = {
        type_: priority
        for priority, type_or_types in enumerate(type_priority_order)
        for type_ in (type_or_types if isinstance(type_or_types, Iterable) else (type_or_types,))
    }

    @deprecated("use `get_item` instead")
    @override
    async def get(self) -> tuple[int, T]:
        return await super().get()

    @deprecated("use `put_item` instead")
    @override
    async def put(self, item: tuple[int, T]) -> None:
        await super().put(item)

    async def put_item(self, item: T) -> None:
        priority = self._type_to_priority.get(type(item), -1)
        # noinspection PyDeprecation
        await self.put((priority, item))

    async def get_item(self) -> T:
        # noinspection PyDeprecation
        _, item = await self.get()
        return item

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        return await self.get_item()
