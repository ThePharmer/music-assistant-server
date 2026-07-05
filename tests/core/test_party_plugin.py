"""Tests for the party plugin (duplicate prevention, guest identity)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType, PlaybackState
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.party import (
    CONF_ENABLE_ADD_QUEUE,
    CONF_ENABLE_BOOST,
    CONF_ENABLE_GUEST_ACCESS,
    CONF_PREVENT_DUPLICATE_TRACKS,
    PARTY_GUEST_USER,
    PartyPlugin,
)


def _create_party_plugin() -> PartyPlugin:
    """Create a minimally configured party plugin for unit tests."""
    plugin = PartyPlugin.__new__(PartyPlugin)
    plugin.mass = MagicMock()
    plugin.mass.music = MagicMock()
    plugin.mass.player_queues = MagicMock()
    plugin.logger = MagicMock()
    plugin.config = MagicMock()
    plugin._queue_lock = asyncio.Lock()
    plugin.get_party_player = AsyncMock(return_value="party_queue")  # type: ignore[method-assign]
    config_values = {
        CONF_ENABLE_GUEST_ACCESS: True,
        CONF_ENABLE_BOOST: True,
        CONF_ENABLE_ADD_QUEUE: True,
        CONF_PREVENT_DUPLICATE_TRACKS: True,
    }
    plugin.config.get_value.side_effect = config_values.__getitem__
    # Back the raw provider config (guest identity secret) with a plain dict
    raw_store: dict[str, str] = {}
    plugin.mass.config.get_raw_provider_config_value.side_effect = lambda _inst, key, default=None: (
        raw_store.get(key, default)
    )
    plugin.mass.config.set_raw_provider_config_value.side_effect = lambda _inst, key, value: (
        raw_store.__setitem__(key, value)
    )
    return plugin


@pytest.mark.asyncio
async def test_add_to_queue_rechecks_duplicates_during_priority_insert() -> None:
    """Reject a duplicate that appears after the initial queue lookup."""
    plugin = _create_party_plugin()
    player_queues = cast("MagicMock", plugin.mass.player_queues)
    music = cast("MagicMock", plugin.mass.music)
    uri = "spotify://track/123"

    queue = MagicMock()
    queue.state = PlaybackState.PLAYING
    queue.current_index = 0
    queue.index_in_buffer = 0
    player_queues.get.return_value = queue
    player_queues.items.return_value = []
    player_queues.load = AsyncMock()

    async def mutate_queue_during_resolve(_uri: str) -> MagicMock:
        media_item = MagicMock()
        media_item.media_type = MediaType.TRACK
        player_queues.items.return_value = [MagicMock(uri=uri, extra_attributes={})]
        return media_item

    music.get_item_by_uri = AsyncMock(side_effect=mutate_queue_during_resolve)
    queue_item = MagicMock()
    queue_item.extra_attributes = {}

    with (
        patch(
            "music_assistant.providers.party.get_current_user",
            return_value=SimpleNamespace(username=PARTY_GUEST_USER),
        ),
        patch("music_assistant.providers.party.QueueItem.from_media_item", return_value=queue_item),
        pytest.raises(InvalidDataError, match="already in the queue"),
    ):
        await plugin.add_to_queue(uri)

    player_queues.load.assert_not_awaited()


def _patch_guest_user() -> Any:
    """Patch get_current_user to return the party guest user."""
    return patch(
        "music_assistant.providers.party.get_current_user",
        return_value=SimpleNamespace(username=PARTY_GUEST_USER),
    )


@pytest.mark.asyncio
async def test_register_guest_returns_verifiable_credentials() -> None:
    """Registered credentials must verify; the name is sanitized and length-capped."""
    plugin = _create_party_plugin()

    with _patch_guest_user():
        result = await plugin.register_guest("  Dance   Commander XXXXXXXXXXXXXXXX ")

    assert result["display_name"] == "Dance Commander XXXXXXXX"
    assert len(result["display_name"]) <= 24
    assert plugin._verify_guest_identity(
        result["guest_id"], result["display_name"], result["guest_sig"]
    )

    with _patch_guest_user():
        fallback = await plugin.register_guest("   ")
    assert fallback["display_name"] == "Guest"


@pytest.mark.asyncio
async def test_register_guest_ids_are_unique_and_stable_across_calls() -> None:
    """Each registration mints a fresh id; the signing secret persists between calls."""
    plugin = _create_party_plugin()

    with _patch_guest_user():
        first = await plugin.register_guest("Alice")
        second = await plugin.register_guest("Alice")

    assert first["guest_id"] != second["guest_id"]
    # Same persisted secret: credentials from the first call still verify
    assert plugin._verify_guest_identity(
        first["guest_id"], first["display_name"], first["guest_sig"]
    )


@pytest.mark.asyncio
async def test_add_to_queue_stamps_guest_identity() -> None:
    """A valid identity is stamped onto the queue item's extra attributes."""
    plugin = _create_party_plugin()
    player_queues = cast("MagicMock", plugin.mass.player_queues)

    queue = MagicMock()
    queue.state = PlaybackState.PLAYING
    player_queues.get.return_value = queue
    plugin._add_to_priority_section = AsyncMock()  # type: ignore[method-assign]

    with _patch_guest_user():
        creds = await plugin.register_guest("Alice")
        await plugin.add_to_queue(
            "spotify://track/123",
            guest_id=creds["guest_id"],
            guest_name=creds["display_name"],
            guest_sig=creds["guest_sig"],
        )

    assert plugin._add_to_priority_section.await_args is not None
    extra_attrs = plugin._add_to_priority_section.await_args.args[2]
    assert extra_attrs["party_guest"] is True
    assert extra_attrs["party_guest_id"] == creds["guest_id"]
    assert extra_attrs["party_guest_name"] == "Alice"


@pytest.mark.asyncio
async def test_add_to_queue_rejects_forged_identity() -> None:
    """A tampered signature or partial identity is rejected before any queue change."""
    plugin = _create_party_plugin()
    plugin._add_to_priority_section = AsyncMock()  # type: ignore[method-assign]

    with _patch_guest_user():
        creds = await plugin.register_guest("Alice")

        # Forged name under someone else's signature
        with pytest.raises(InvalidDataError, match="Invalid guest identity"):
            await plugin.add_to_queue(
                "spotify://track/123",
                guest_id=creds["guest_id"],
                guest_name="Mallory",
                guest_sig=creds["guest_sig"],
            )

        # Partial identity (id without signature)
        with pytest.raises(InvalidDataError, match="Invalid guest identity"):
            await plugin.add_to_queue(
                "spotify://track/123",
                guest_id=creds["guest_id"],
            )

    plugin._add_to_priority_section.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_to_queue_without_identity_still_works() -> None:
    """Identity-less calls (stock frontend) remain accepted, just unattributed."""
    plugin = _create_party_plugin()
    player_queues = cast("MagicMock", plugin.mass.player_queues)

    queue = MagicMock()
    queue.state = PlaybackState.PLAYING
    player_queues.get.return_value = queue
    plugin._add_to_priority_section = AsyncMock()  # type: ignore[method-assign]

    with _patch_guest_user():
        await plugin.add_to_queue("spotify://track/123")

    assert plugin._add_to_priority_section.await_args is not None
    extra_attrs = plugin._add_to_priority_section.await_args.args[2]
    assert extra_attrs["party_guest"] is True
    assert "party_guest_id" not in extra_attrs
