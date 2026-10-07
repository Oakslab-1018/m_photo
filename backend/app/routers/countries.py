"""Country knowledge-base endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import Settings
from ..deps import get_settings
from ..schemas import CountriesResponse
from ..services.countries import CountriesService

router = APIRouter(prefix="/geo", tags=["geo"])


@router.get("/countries", response_model=CountriesResponse)
def list_countries(settings: Settings = Depends(get_settings)) -> CountriesResponse:
    service = CountriesService(settings)
    return CountriesResponse(countries=service.list_countries())


__all__ = ["router"]
