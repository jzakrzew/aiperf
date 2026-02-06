# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import ValidationError

from aiperf.common import random_generator as rng
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import MediaType
from aiperf.common.models import Conversation, Turn
from aiperf.common.types import MediaTypeT
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import RandomPool
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy

# Type aliases
Filename: TypeAlias = str
# Mapping from media type to named content pools: {name: [content_strings]}
ContentPools: TypeAlias = dict[MediaTypeT, dict[str, list[str]]]


class RandomPoolDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads data from a single file or a directory.

    Each line in the file represents single-turn conversation data,
    and files create individual pools for random sampling:
      - Single file: All lines form one single pool (to be randomly sampled from)
      - Directory: Each file becomes a separate pool, then pools are randomly sampled
                   and merged into conversations later.

    The random pool custom dataset
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch size > 1)
      - supports named fields for each modality (e.g. text_field_a, text_field_b, etc.)
      - DOES NOT support multi-turn or its features (e.g. delay, sessions, etc.)

    Example:

    1. Single file
    ```jsonl
    {"text": "Who are you?", "image": "/path/to/image1.png"}
    {"text": "Explain what is the meaning of life.", "image": "/path/to/image2.png"}
    ...
    ```
    The file will form a single pool of text and image data that will be used
    to generate conversations.

    2. Directory

    Directory will be useful if user wants to
      - create multiple pools of different modalities separately (e.g. text, image)
      - specify different field names for the same modality.

    data/queries.jsonl
    ```jsonl
    {"texts": [{"name": "query", "contents": ["Who are you?"]}]}
    {"texts": [{"name": "query", "contents": ["What is the meaning of life?"]}]}
    ...
    ```

    data/passages.jsonl
    ```jsonl
    {"texts": [{"name": "passage", "contents": ["I am a cat."]}]}
    {"texts": [{"name": "passage", "contents": ["I am a dog."]}]}
    ...
    ```

    The loader will create two separate pools for each file: queries and passages.
    Each pool is a text dataset with a different field name (e.g. query, passage),
    and loader will later sample from these two pools to create conversations.
    """

    def __init__(
        self,
        *,
        filename: str,
        user_config: UserConfig,
        num_conversations: int = 1,
        **kwargs,
    ):
        super().__init__(filename=filename, user_config=user_config, **kwargs)
        self._rng = rng.derive("dataset.loader.random_pool")
        self.num_conversations = num_conversations

    @staticmethod
    def _validate_path(path: Path) -> int:
        """Validate all files and directories recursively against the RandomPool model.

        Args:
            path: The path to the file or directory to validate.

        Returns:
            int: Count of files with at least one valid line.

        Raises:
            ValidationError: If any file contains invalid data.
        """
        valid_count = 0

        if path.is_dir():
            # if path is a directory, recursively call this function for each child
            # if any child fails validation, it will exit early with an exception
            for file in path.iterdir():
                valid_count += RandomPoolDatasetLoader._validate_path(file)

        elif path.is_file():
            # if path is a file, validate the first non-empty line against the RandomPool model
            # if the line is valid, increment the valid count and break the loop,
            # otherwise a ValidationError will be raised and the function will exit early
            with open(path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    RandomPool.model_validate_json(line)
                    valid_count += 1
                    break

        return valid_count

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        RandomPool is the only loader that supports directory inputs.
        For structural detection, RandomPool format is ambiguous with SingleTurn
        (both have modality fields), so explicit 'type' field or directory path is required.

        Returns:
            True only if filename is a directory with at least one valid file.
            False otherwise (including for regular files without explicit type).
        """

        if data is not None and data.get("type") == CustomDatasetType.RANDOM_POOL:
            try:
                RandomPool.model_validate(data)
                return True
            except ValidationError:
                return False

        if filename is not None:
            try:
                path = Path(filename) if isinstance(filename, str) else filename
                # Only match directories - files are ambiguous with SingleTurn
                if path.is_dir():
                    valid_count = cls._validate_path(path)
                    return valid_count > 0
                return False
            except ValidationError:
                return False

        # RandomPool schema is very similar to SingleTurn, so we can't reliably
        # distinguish without an explicit type field or directory path
        return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for RandomPool."""
        return DatasetSamplingStrategy.SHUFFLE

    def load_dataset(self) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from a file or directory.

        If filename is a file, reads and parses using the RandomPool model.
        If filename is a directory, reads each file in the directory and merges
        items with different modality names into combined RandomPool objects.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        path = Path(self.filename)

        if path.is_file():
            dataset_pool = self._load_dataset_from_file(path)
            return {path.name: dataset_pool}

        return self._load_dataset_from_dir(path)

    def _load_dataset_from_file(self, file_path: Path) -> list[RandomPool]:
        """Load random pool data from a single file.

        Args:
            file_path: The path to the file containing the data.

        Returns:
            A list of RandomPool objects.
        """
        dataset_pool: list[RandomPool] = []

        with open(file_path) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                random_pool_data = RandomPool.model_validate_json(line)
                dataset_pool.append(random_pool_data)

        return dataset_pool

    def _load_dataset_from_dir(
        self, dir_path: Path
    ) -> dict[Filename, list[RandomPool]]:
        """Load random pool data from all files in a directory.

        Args:
            dir_path: The path to the directory containing the files.

        Returns:
            A dictionary mapping filename to list of RandomPool objects.
        """
        data: dict[Filename, list[RandomPool]] = defaultdict(list)

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file():
                dataset_pool = self._load_dataset_from_file(file_path)
                data[file_path.name].extend(dataset_pool)

        return data

    def convert_to_conversations(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert random pool data to conversation objects.

        Each RandomPool entry becomes a single-turn conversation with a unique session ID.
        When any modality batch_size > 1, switches to content-level sampling where
        individual contents are randomly sampled from the pool (matching genai-perf
        behavior for embedding/ranking endpoints).

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations.
        """
        if self._has_batch_override():
            return self._convert_with_batching(data)
        return self._convert_default(data)

    def _has_batch_override(self) -> bool:
        """Check if any modality batch_size exceeds the default of 1.

        Returns:
            True if any batch_size > 1, indicating content-level batching is needed.
        """
        cfg = self.user_config.input
        return any(
            bs > 1
            for bs in [
                cfg.prompt.batch_size,
                cfg.image.batch_size,
                cfg.audio.batch_size,
                cfg.video.batch_size,
            ]
        )

    def _get_batch_size(self, modality: MediaTypeT) -> int:
        """Get the configured batch size for a given modality.

        Args:
            modality: The media type (e.g. MediaType.TEXT, MediaType.IMAGE).

        Returns:
            The batch size for the modality.
        """
        cfg = self.user_config.input
        batch_size_map: dict[MediaTypeT, int] = {
            MediaType.TEXT: cfg.prompt.batch_size,
            MediaType.IMAGE: cfg.image.batch_size,
            MediaType.AUDIO: cfg.audio.batch_size,
            MediaType.VIDEO: cfg.video.batch_size,
        }
        return batch_size_map[modality]

    def _build_content_pools(
        self, entries: list[RandomPool], default_name: str
    ) -> ContentPools:
        """Extract and flatten all contents from pool entries, grouped by (modality, name).

        Uses MediaConversionMixin.convert_to_media_objects() to normalize all entry
        formats (singular, plural, named objects) into Media objects, then flattens
        their contents into pools keyed by name.

        Args:
            entries: The RandomPool entries from a single file.
            default_name: The default name for unnamed media fields (typically filename stem).

        Returns:
            A nested dict: {media_type: {name: [content_strings]}}.
        """
        pools: ContentPools = {
            media_type: defaultdict(list)
            for media_type in [
                MediaType.TEXT,
                MediaType.IMAGE,
                MediaType.AUDIO,
                MediaType.VIDEO,
            ]
        }

        for entry in entries:
            media = self.convert_to_media_objects(entry, name=default_name)
            for media_type, media_objects in media.items():
                for media_obj in media_objects:
                    pools[media_type][media_obj.name].extend(media_obj.contents)

        return pools

    def _validate_batch_sizes(self, pools: ContentPools, filename: str) -> None:
        """Validate that batch sizes don't exceed available pool sizes.

        Args:
            pools: The content pools built from a file.
            filename: The source filename (for error messages).

        Raises:
            ValueError: If any batch_size exceeds the pool size for a modality.
        """
        for media_type, named_pools in pools.items():
            batch_size = self._get_batch_size(media_type)
            for name, contents in named_pools.items():
                if batch_size > len(contents):
                    field_desc = f"{media_type} (name={name!r})" if name else media_type
                    raise ValueError(
                        f"Batch size {batch_size} for {field_desc} exceeds the number "
                        f"of available items ({len(contents)}) in pool '{filename}'. "
                        f"Provide more data or reduce --batch-size-{media_type}."
                    )

    def _convert_with_batching(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert pool data using content-level sampling for batched requests.

        Instead of sampling whole RandomPool entries, this flattens all contents
        per modality into pools and samples batch_size items (without replacement)
        for each conversation. This matches genai-perf behavior for embedding and
        ranking endpoints where each request carries multiple inputs.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations.
        """
        from aiperf.common.models import Audio, Image, Text, Video

        media_class_map = {
            MediaType.TEXT: Text,
            MediaType.IMAGE: Image,
            MediaType.AUDIO: Audio,
            MediaType.VIDEO: Video,
        }

        conversations = [
            Conversation(session_id=self.session_id_generator.next())
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        for filename, dataset_pool in data.items():
            default_name = Path(filename).stem
            pools = self._build_content_pools(dataset_pool, default_name)
            self._validate_batch_sizes(pools, filename)

            turns: list[Turn] = []
            for _ in range(self.num_conversations):
                turn_media: dict[MediaTypeT, list] = {
                    MediaType.TEXT: [],
                    MediaType.IMAGE: [],
                    MediaType.AUDIO: [],
                    MediaType.VIDEO: [],
                }
                for media_type, named_pools in pools.items():
                    batch_size = self._get_batch_size(media_type)
                    media_cls = media_class_map[media_type]
                    for name, contents in named_pools.items():
                        sampled = self._rng.sample(contents, k=batch_size)
                        turn_media[media_type].append(
                            media_cls(name=name, contents=sampled)
                        )
                turns.append(
                    Turn(
                        texts=turn_media[MediaType.TEXT],
                        images=turn_media[MediaType.IMAGE],
                        audios=turn_media[MediaType.AUDIO],
                        videos=turn_media[MediaType.VIDEO],
                    )
                )
            sampled_dataset[filename] = turns

        # Merge turns for each conversation (same as default path)
        for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
            turn = self._merge_turns(batched_turns)
            conversations[i].turns.append(turn)

        return conversations

    def _convert_default(
        self, data: dict[Filename, list[RandomPool]]
    ) -> list[Conversation]:
        """Convert pool data using default entry-level sampling.

        Each conversation gets one randomly sampled RandomPool entry per file pool.
        This preserves cross-modality correlations within entries.

        Args:
            data: A dictionary mapping filename to list of RandomPool objects.

        Returns:
            A list of conversations.
        """
        conversations = [
            Conversation(session_id=self.session_id_generator.next())
            for _ in range(self.num_conversations)
        ]

        # F x N (F: num of files, N: num of conversations)
        sampled_dataset: dict[Filename, list[Turn]] = {}

        # Randomly sample (with replacement) from each dataset pool
        for filename, dataset_pool in data.items():
            samples = self._rng.choices(dataset_pool, k=self.num_conversations)
            turns: list[Turn] = []
            for sample in samples:
                media = self.convert_to_media_objects(sample, name=Path(filename).stem)
                turns.append(
                    Turn(
                        texts=media[MediaType.TEXT],
                        images=media[MediaType.IMAGE],
                        audios=media[MediaType.AUDIO],
                        videos=media[MediaType.VIDEO],
                    )
                )
            sampled_dataset[filename] = turns

        # Merge turns for each conversation
        for i, batched_turns in enumerate(zip(*sampled_dataset.values(), strict=False)):
            turn = self._merge_turns(batched_turns)
            conversations[i].turns.append(turn)

        return conversations

    def _merge_turns(self, turns: list[Turn]) -> Turn:
        """Merge turns into a single turn.

        Args:
            turns: A list of turns.

        Returns:
            A single turn.
        """
        merged_turn = Turn(
            texts=[text for turn in turns for text in turn.texts],
            images=[image for turn in turns for image in turn.images],
            audios=[audio for turn in turns for audio in turn.audios],
            videos=[video for turn in turns for video in turn.videos],
        )
        return merged_turn
