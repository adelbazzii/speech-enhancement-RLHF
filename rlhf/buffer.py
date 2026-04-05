class ExperienceBuffer:
    """Collects old RL policy and related data for PPO update."""

    def __init__(self):
        self._buffer = []

    def add(self, experience: dict):
        self._buffer.append(experience)

    def __len__(self):
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def clear(self):
        self._buffer = []
