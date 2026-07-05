"""Tests for the music controller."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.media_items import SearchResults

from music_assistant.constants import VACUUM_MIN_RECLAIM_RATIO
from music_assistant.controllers.music import MusicController
from music_assistant.helpers.database import DatabaseConnection
from music_assistant.mass import MusicAssistant


@pytest.fixture
async def music(mass_minimal: MusicAssistant) -> AsyncGenerator[MusicController]:
    """Return a music controller attached to the minimal mass instance."""
    controller = MusicController(mass_minimal)
    mass_minimal.music = controller
    yield controller
    # close the db connection so its worker thread does not outlive the test
    if controller._database:
        await controller._database.close()


async def test_setup_skips_vacuum_when_little_reclaimable(music: MusicController) -> None:
    """Test that the library db startup vacuum is skipped when little can be reclaimed."""
    with (
        patch.object(
            DatabaseConnection,
            "get_reclaimable_ratio",
            AsyncMock(return_value=VACUUM_MIN_RECLAIM_RATIO / 2),
        ),
        patch.object(DatabaseConnection, "vacuum", AsyncMock()) as mock_vacuum,
    ):
        await music._setup_database()
    mock_vacuum.assert_not_called()


async def test_setup_runs_vacuum_when_reclaimable(music: MusicController) -> None:
    """Test that the library db startup vacuum runs when enough space can be reclaimed."""
    with (
        patch.object(
            DatabaseConnection,
            "get_reclaimable_ratio",
            AsyncMock(return_value=VACUUM_MIN_RECLAIM_RATIO + 0.1),
        ),
        patch.object(DatabaseConnection, "vacuum", AsyncMock()) as mock_vacuum,
    ):
        await music._setup_database()
    mock_vacuum.assert_awaited_once_with()


async def test_search_provider_failure_degrades_gracefully(music: MusicController) -> None:
    """Test that a failing provider yields empty results instead of failing the whole search.

    Rebase tripwire: upstream's stock `_search_provider` re-raises provider errors, which
    lets a single broken provider return zero results for ALL providers (support#5209
    failure shape). If an upstream rebase reverts our graceful-degradation change in
    `controllers/music.py`, this test fails.
    """
    prov = MagicMock()
    prov.name = "Broken Provider"
    prov.supported_features = {ProviderFeature.SEARCH}
    prov.search = AsyncMock(side_effect=RuntimeError("simulated provider outage"))
    with patch.object(music.mass, "get_provider", return_value=prov):
        results = await music._search_provider("test query", "broken", [MediaType.TRACK])
    assert isinstance(results, SearchResults)
    assert not results.tracks
    assert not results.artists


async def test_search_provider_none_result_degrades_gracefully(music: MusicController) -> None:
    """Test that a provider returning None yields empty results instead of an error."""
    prov = MagicMock()
    prov.name = "Stub Provider"
    prov.supported_features = {ProviderFeature.SEARCH}
    prov.search = AsyncMock(return_value=None)
    with patch.object(music.mass, "get_provider", return_value=prov):
        results = await music._search_provider("test query", "stub", [MediaType.TRACK])
    assert isinstance(results, SearchResults)
    assert not results.tracks
