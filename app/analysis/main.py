import asyncio

from app.core import Paths
from app.model import EmulationPrediction, EmulationState

from .predictions import infer_last_window_from_lists


def predict_sync(emulation_state: EmulationState) -> EmulationPrediction:
    messages, details = infer_last_window_from_lists(
        bpm_pairs=(
            emulation_state.part_data_log.bpm
            + emulation_state.shifted_plot(emulation_state.sent_part_data.bpm)
        ),
        uc_pairs=(
            emulation_state.part_data_log.uterus
            + emulation_state.shifted_plot(emulation_state.sent_part_data.uterus)
        ),
        ckpt_path=Paths.ml.predictor_model,
    )

    return EmulationPrediction(
        messages=messages,
        bpm_average=details["fhr_next60_mean"],
        bpm_min=details["fhr_next60_min"],
    )


async def predict(emulation_state: EmulationState) -> EmulationPrediction:
    return await asyncio.to_thread(predict_sync, emulation_state)
