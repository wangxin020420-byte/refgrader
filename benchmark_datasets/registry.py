from __future__ import annotations

from .adapters import prepare_asap_sas, prepare_mohler, prepare_sas_bench


ADAPTERS = {
    "asap_sas": prepare_asap_sas,
    "mohler": prepare_mohler,
    "sas_bench": prepare_sas_bench,
}


def get_adapter(dataset_name: str):
    try:
        return ADAPTERS[str(dataset_name).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown public dataset adapter: {dataset_name}. "
            f"Available: {', '.join(sorted(ADAPTERS))}"
        ) from exc
