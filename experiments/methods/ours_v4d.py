"""memory_v4 + distillation: same as ours_v4 but with knowledge distillation enabled at finalize."""

from .ours_v4 import OursV4


class OursV4D(OursV4):
    name = "ours_v4d"

    def finalize(self) -> None:
        self.agent.build_summaries(force=True, distill=True)
