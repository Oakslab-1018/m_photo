"""Bounded-provider transport tests.

One logical lookup may retry transient network failures (a congested link can
drop the first TLS handshake), but retries never leak into HTTP status
handling and only one failure cooldown is recorded after the final attempt.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.geocoding import (  # noqa: E402
    MAPTILER_DEFAULT_URL,
    GeocoderRequest,
    GeocodingConfig,
    GeocodingUnavailable,
    MapTilerGeocoder,
    PhotonGeocoder,
)

_PHOTON_PAYLOAD = {
    "features": [
        {
            "properties": {
                "name": "Berlin",
                "city": "Berlin",
                "countrycode": "DE",
                "osm_id": 240109189,
                "osm_type": "N",
            }
        }
    ]
}

_MAPTILER_PAYLOAD = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": "subregion.537",
            "text": "Chengdu",
            "place_name": "Chengdu, Sichuan, China",
            "properties": {"ref": "osm:r2110264", "country_code": "cn"},
            "context": [
                {"id": "region.193", "text": "Sichuan", "country_code": "cn"},
                {"id": "country.134", "text": "China", "country_code": "cn"},
            ],
        }
    ],
}


class _NoWaitLimiter:
    def wait(self) -> None:
        return


def _geocoder(
    transport: httpx.BaseTransport,
    **overrides: object,
) -> PhotonGeocoder:
    config = GeocodingConfig(
        timeout_seconds=4.0,
        connect_timeout_seconds=1.5,
        retry_backoff_seconds=0.0,
        min_interval_seconds=1.0,
        **overrides,
    )
    geocoder = PhotonGeocoder(config, transport=transport)
    # The suite must not sleep for the public-provider rate limit.
    geocoder._limiter = _NoWaitLimiter()  # type: ignore[assignment]
    return geocoder


class GeocodingRetryTest(unittest.TestCase):
    def test_transient_connect_timeout_is_retried_within_one_lookup(self):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 3:
                raise httpx.ConnectTimeout("simulated handshake timeout", request=request)
            return httpx.Response(200, json=_PHOTON_PAYLOAD)

        geocoder = _geocoder(httpx.MockTransport(handler))

        hits = geocoder.search(GeocoderRequest(terms=("Berlin",)))

        self.assertEqual(len(calls), 3)
        self.assertEqual(hits[0].name, "Berlin")
        # A success must not leave the provider in a failure cooldown.
        self.assertEqual(geocoder._cooldown_until, 0.0)

    def test_retries_are_bounded_and_cooldown_starts_after_the_last_attempt(self):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ConnectTimeout("simulated handshake timeout", request=request)

        geocoder = _geocoder(httpx.MockTransport(handler), max_attempts=2)

        with self.assertRaises(GeocodingUnavailable) as first:
            geocoder.search(GeocoderRequest(terms=("Berlin",)))
        self.assertEqual(first.exception.reason, "connect_timeout")
        self.assertTrue(first.exception.retryable)
        self.assertEqual(len(calls), 2)

        # The cooldown is recorded once after retries are exhausted; a new
        # lookup is suppressed without opening another socket.
        with self.assertRaises(GeocodingUnavailable) as second:
            geocoder.search(GeocoderRequest(terms=("Munich",)))
        self.assertTrue(second.exception.cooldown)
        self.assertEqual(second.exception.reason, "connect_timeout")
        self.assertEqual(len(calls), 2)

    def test_http_status_errors_are_not_retried(self):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, json={"error": "bad request"})

        geocoder = _geocoder(httpx.MockTransport(handler))

        with self.assertRaises(GeocodingUnavailable) as ctx:
            geocoder.search(GeocoderRequest(terms=("Berlin",)))

        self.assertEqual(ctx.exception.reason, "http_status")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(len(calls), 1)

    def test_single_attempt_configuration_disables_retries(self):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ConnectTimeout("simulated handshake timeout", request=request)

        geocoder = _geocoder(httpx.MockTransport(handler), max_attempts=1)

        with self.assertRaises(GeocodingUnavailable):
            geocoder.search(GeocoderRequest(terms=("Berlin",)))

        self.assertEqual(len(calls), 1)


class MapTilerGeocoderTest(unittest.TestCase):
    @staticmethod
    def _geocoder(handler) -> MapTilerGeocoder:
        config = GeocodingConfig(
            provider="maptiler",
            base_url=MAPTILER_DEFAULT_URL,
            api_key="test-key",
            referer="http://127.0.0.1:8080/",
            timeout_seconds=4.0,
            connect_timeout_seconds=1.5,
            retry_backoff_seconds=0.0,
            min_interval_seconds=1.0,
        )
        geocoder = MapTilerGeocoder(config, transport=httpx.MockTransport(handler))
        geocoder._limiter = _NoWaitLimiter()  # type: ignore[assignment]
        return geocoder

    def test_search_sends_key_referer_and_country_and_parses_locality(self):
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json=_MAPTILER_PAYLOAD)

        hits = self._geocoder(handler).search(
            GeocoderRequest(terms=("Chengdu",), country_iso="CN")
        )

        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.name, "Chengdu")
        self.assertEqual(hit.display_name, "Chengdu, Sichuan, China")
        self.assertEqual(hit.provider_place_id, "subregion.537")
        self.assertEqual(hit.country_code, "CN")
        self.assertEqual(hit.admin1_name, "Sichuan")
        self.assertIn("Sichuan", hit.locality_names)
        self.assertIn("China", hit.locality_names)

        request = calls[0]
        self.assertEqual(request.url.path, "/geocoding/Chengdu.json")
        self.assertEqual(request.url.params["key"], "test-key")
        self.assertEqual(request.url.params["country"], "cn")
        self.assertEqual(request.headers["Referer"], "http://127.0.0.1:8080/")

    def test_malformed_payload_is_reported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        with self.assertRaises(GeocodingUnavailable) as ctx:
            self._geocoder(handler).search(GeocoderRequest(terms=("Chengdu",)))

        self.assertEqual(ctx.exception.reason, "malformed_response")


class GeocodingConfigTest(unittest.TestCase):
    def test_attempt_settings_are_read_from_env(self):
        config = GeocodingConfig.from_env({
            "M_PHOTO_GEOCODER_MAX_ATTEMPTS": "4",
            "M_PHOTO_GEOCODER_RETRY_BACKOFF_SECONDS": "0",
        })

        self.assertEqual(config.max_attempts, 4)
        self.assertEqual(config.retry_backoff_seconds, 0.0)

    def test_maptiler_requires_a_key_and_an_explicit_url(self):
        enabled = GeocodingConfig.from_env({
            "M_PHOTO_GEOCODER_PROVIDER": "maptiler",
            "M_PHOTO_GEOCODER_BASE_URL": MAPTILER_DEFAULT_URL,
            "M_PHOTO_GEOCODER_API_KEY": "test-key",
            "M_PHOTO_GEOCODER_REFERER": "http://127.0.0.1:8080/",
        })
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.base_url, MAPTILER_DEFAULT_URL)
        self.assertEqual(enabled.referer, "http://127.0.0.1:8080/")

        without_url = GeocodingConfig.from_env({
            "M_PHOTO_GEOCODER_PROVIDER": "maptiler",
            "M_PHOTO_GEOCODER_API_KEY": "test-key",
        })
        self.assertFalse(without_url.enabled)

        without_key = GeocodingConfig.from_env({
            "M_PHOTO_GEOCODER_PROVIDER": "maptiler",
            "M_PHOTO_GEOCODER_BASE_URL": MAPTILER_DEFAULT_URL,
        })
        self.assertFalse(without_key.enabled)

    def test_attempt_settings_are_bounded(self):
        self.assertEqual(
            GeocodingConfig.from_env({"M_PHOTO_GEOCODER_MAX_ATTEMPTS": "99"}).max_attempts,
            5,
        )
        self.assertEqual(
            GeocodingConfig.from_env({"M_PHOTO_GEOCODER_MAX_ATTEMPTS": "0"}).max_attempts,
            1,
        )
        self.assertEqual(
            GeocodingConfig.from_env({"M_PHOTO_GEOCODER_MAX_ATTEMPTS": "not-a-number"}).max_attempts,
            3,
        )
        self.assertEqual(
            GeocodingConfig.from_env(
                {"M_PHOTO_GEOCODER_RETRY_BACKOFF_SECONDS": "99"}
            ).retry_backoff_seconds,
            5.0,
        )
        self.assertEqual(
            GeocodingConfig.from_env(
                {"M_PHOTO_GEOCODER_KEEPALIVE_SECONDS": "9999"}
            ).keepalive_seconds,
            600.0,
        )
        self.assertEqual(
            GeocodingConfig.from_env(
                {"M_PHOTO_GEOCODER_KEEPALIVE_SECONDS": "0"}
            ).keepalive_seconds,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
