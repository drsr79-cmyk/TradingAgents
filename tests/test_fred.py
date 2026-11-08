"""FRED macro vendor: alias resolution, configuration errors, output formatting,
missing-value handling, lookahead-safe windowing, and router integration.

All API access is mocked, so these run without a network connection or a key.
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors import fred

# A small, stable set of observations to format against.
_META = {
    "seriess": [
        {
            "title": "Unemployment Rate",
            "units_short": "%",
            "frequency": "Monthly",
            "seasonal_adjustment_short": "SA",
        }
    ]
}
_OBS = {
    "observations": [
        {"date": "2025-06-01", "value": "4.1"},
        {"date": "2025-07-01", "value": "4.3"},
        {"date": "2025-08-01", "value": "."},   # missing -> skipped
        {"date": "2025-09-01", "value": "4.4"},
    ]
}


def _request_stub(meta=_META, obs=_OBS):
    """Build a _request replacement that dispatches on the endpoint path."""
    def _impl(path, params):
        if path == "series":
            return meta
        if path == "series/observations":
            return obs
        raise AssertionError(f"unexpected FRED path: {path}")
    return _impl


@pytest.mark.unit
class FredResolutionTests(unittest.TestCase):
    def test_alias_maps_to_series_id(self):
        self.assertEqual(fred._resolve_series_id("cpi"), "CPIAUCSL")
        self.assertEqual(fred._resolve_series_id("unemployment"), "UNRATE")

    def test_alias_is_case_and_separator_insensitive(self):
        self.assertEqual(fred._resolve_series_id("Fed Funds Rate"), "FEDFUNDS")
        self.assertEqual(fred._resolve_series_id("10y-treasury"), "DGS10")

    def test_euro_area_aliases_map_to_series_ids(self):
        # Euro-area context for non-US tickers (e.g. Euronext .PA): FRED mirrors
        # the ECB policy rates, Eurostat HICP/GDP and OECD long-term yields.
        expected = {
            "ecb_deposit_rate": "ECBDFR",
            "ecb_main_refi_rate": "ECBMRRFR",
            "euro_hicp": "CP0000EZ19M086NEST",
            "euro_core_hicp": "00XEFDEZ19M086NEST",
            "euro_real_gdp": "CLVMNACSCAB1GQEA19",
            "germany_10y": "IRLTLT01DEM156N",
            "france_10y": "IRLTLT01FRM156N",
            "eur_usd": "DEXUSEU",
        }
        for alias, series_id in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(fred._resolve_series_id(alias), series_id)
        self.assertEqual(fred._resolve_series_id("ECB Deposit Rate"), "ECBDFR")
        self.assertEqual(fred._resolve_series_id("euro-hicp"), "CP0000EZ19M086NEST")

    def test_unknown_alias_is_treated_as_raw_series_id(self):
        # Power users can pass any FRED series ID; we uppercase by convention.
        self.assertEqual(fred._resolve_series_id("dgs30"), "DGS30")
        self.assertEqual(fred._resolve_series_id("MyCustomSeries"), "MYCUSTOMSERIES")

    def test_descriptive_phrase_is_rejected(self):
        # An LLM phrase (spaces / too long) is not a series ID — reject up front
        # with guidance rather than 400ing the API.
        for bad in ("bank of japan rate", "the unemployment number", "X" * 31):
            with self.assertRaises(ValueError):
                fred._resolve_series_id(bad)

    def test_get_macro_data_returns_guidance_on_bad_indicator(self):
        # Invalid indicator -> actionable message, not a crash (no API call).
        out = fred.get_macro_data("bank of japan rate", "2026-01-01")
        self.assertIn("FRED", out)
        self.assertIn("not a known macro alias", out)


@pytest.mark.unit
class FredConfigTests(unittest.TestCase):
    def test_missing_key_raises_not_configured(self):
        from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

        with mock.patch.dict("os.environ", {}, clear=True), \
                self.assertRaises(AnalysisProcessHasNoVendorKeys):
            fred.get_api_key()

    def test_not_configured_is_a_value_error(self):
        # Routing relies on this subclassing for "vendor unavailable" handling.
        self.assertTrue(issubclass(fred.FredNotConfiguredError, ValueError))


@pytest.mark.unit
class FredFormattingTests(unittest.TestCase):
    def test_report_has_header_latest_change_and_table(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        self.assertIn("## FRED: Unemployment Rate (UNRATE)", out)
        self.assertIn("Units: %", out)
        self.assertIn("Frequency: Monthly (SA)", out)
        self.assertIn("**Latest:** 4.4 (2025-09-01)", out)
        # The change names the observations it spans, not the lookback window,
        # so a 3-month move on a monthly series cannot read as year on year.
        self.assertIn("**Change from 2025-06-01 to 2025-09-01:** +0.30 (+7.32%), from 4.1", out)
        self.assertNotIn("Change over window", out)
        self.assertIn("| 2025-06-01 | 4.1 |", out)

    def test_missing_value_is_skipped(self):
        with mock.patch.object(fred, "_request", side_effect=_request_stub()):
            out = fred.get_macro_data("unemployment", "2025-09-30", 365)
        # the "." observation must not appear as a row
        self.assertNotIn("2025-08-01", out)

    def test_empty_window_reports_no_observations(self):
        empty = {"observations": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=empty)):
            out = fred.get_macro_data("unemployment", "2025-09-30", 30)
        self.assertIn("No observations", out)

    def test_unknown_series_returns_not_found_message(self):
        # A well-formed but unknown series ID returns guidance, not a crash, so
        # the run is not aborted over an optional macro lookup.
        no_series = {"seriess": []}
        with mock.patch.object(fred, "_request", side_effect=_request_stub(meta=no_series)):
            out = fred.get_macro_data("totally_unknown_xyz", "2025-09-30", 30)
        self.assertIn("not found", out)

    def test_long_series_is_truncated_but_change_uses_full_range(self):
        # Build > MAX_ROWS observations deterministically.
        obs = {
            "observations": [
                {"date": f"2025-01-{(i % 28) + 1:02d}", "value": str(i)}
                for i in range(fred.MAX_ROWS + 10)
            ]
        }
        with mock.patch.object(fred, "_request", side_effect=_request_stub(obs=obs)):
            out = fred.get_macro_data("unemployment", "2025-12-31", 365)
        self.assertIn(f"most recent {fred.MAX_ROWS}", out)
        # the change must reference the true first (0) and last value
        self.assertIn("+49.00, from 0\n", out)
        body_rows = [ln for ln in out.splitlines() if ln.startswith("| 2025")]
        self.assertEqual(len(body_rows), fred.MAX_ROWS)

    def test_window_is_lookahead_safe(self):
        # observation_end must equal as_of_date so a past date never pulls future data.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("unemployment", "2025-09-30", 90)
        obs_params = captured["series/observations"]
        self.assertEqual(obs_params["observation_end"], "2025-09-30")
        self.assertEqual(obs_params["observation_start"], "2025-07-02")  # 90d back

    def test_requests_pin_the_data_vintage(self):
        # #1275: both the metadata and observations requests must pin the vintage
        # to as_of_date (clamped to FRED's today), or FRED serves the latest
        # revision and revision-prone series leak future information. A past
        # as_of_date sits below FRED's today, so it pins through unchanged.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-01-01"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2025-09-30", 90)

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2025-09-30", path)
            self.assertEqual(captured[path]["realtime_end"], "2025-09-30", path)

    def test_future_curr_date_clamps_vintage_to_fred_today(self):
        # #1275 regression: on a live run as_of_date is the caller's LOCAL date,
        # which can be a day ahead of FRED's US-Central clock. Pinning the vintage
        # to that future date 400s, and the routing layer then drops macro data
        # silently. The pin must clamp to FRED's today; the observation window
        # (future bars can't exist yet) stays at as_of_date.
        captured = {}

        def _capture(path, params):
            captured[path] = params
            return _META if path == "series" else _OBS

        with mock.patch.object(fred, "_fred_today", return_value="2026-08-31"), \
                mock.patch.object(fred, "_request", side_effect=_capture):
            fred.get_macro_data("cpi", "2026-09-01", 90)  # local a day ahead of Chicago

        for path in ("series", "series/observations"):
            self.assertEqual(captured[path]["realtime_start"], "2026-08-31", path)
            self.assertEqual(captured[path]["realtime_end"], "2026-08-31", path)
        # the observation window still tracks as_of_date, not the clamped vintage
        self.assertEqual(captured["series/observations"]["observation_end"], "2026-09-01")


@pytest.mark.unit
class FredRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_macro_category_routes_to_fred(self):
        self.assertEqual(
            router.get_category_for_method("get_macro_indicators"), "macro_data"
        )
        set_config({"data_vendors": {"macro_data": "fred"}})
        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": lambda *a, **k: "MACRO_OK"}},
            clear=False,
        ):
            out = router.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertEqual(out, "MACRO_OK")

    def test_not_configured_degrades_gracefully(self):
        # macro_data is optional: with only fred and no key, the router degrades
        # to a sentinel instead of aborting the run — a missing optional key must
        # not crash an analysis.
        set_config({"data_vendors": {"macro_data": "fred"}})

        def _unconfigured(*a, **k):
            raise fred.FredNotConfiguredError("FRED_API_KEY not set")

        with mock.patch.dict(
            router.VENDOR_METHODS,
            {"get_macro_indicators": {"fred": _unconfigured}},
            clear=False,
        ):
            out = router.route_to_vendor("get_macro_indicators", "cpi", "2026-06-01", 365)
        self.assertIn("DATA_UNAVAILABLE", out)


if __name__ == "__main__":
    unittest.main()


_KEY = "abcdef0123456789abcdef0123456789"


@pytest.mark.unit
class TestKeyKeptOutOfErrors:
    """The analysis process never places the FRED key on a request."""

    def test_http_error_message_carries_no_key(self):
        from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

        called = []
        with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
             mock.patch("tradingagents.dataflows.net.requests.get",
                        side_effect=lambda *a, **k: called.append(a)), \
             pytest.raises(AnalysisProcessHasNoVendorKeys) as caught:
            fred._request("series", {"series_id": "DGS10"})
        assert _KEY not in str(caught.value)
        assert called == []

    def test_connection_error_before_any_response_carries_no_key(self):
        from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

        called = []
        with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
             mock.patch("tradingagents.dataflows.net.requests.get",
                        side_effect=lambda *a, **k: called.append(a)), \
             pytest.raises(AnalysisProcessHasNoVendorKeys) as caught:
            fred._request("series", {"series_id": "DGS10"})
        assert _KEY not in str(caught.value)
        assert called == []


@pytest.mark.unit
def test_error_without_the_key_in_its_message_still_drops_the_request():
    """A prepared request that embeds the key is never sent by this process."""
    from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

    called = []
    with mock.patch.dict("os.environ", {"FRED_API_KEY": _KEY}), \
         mock.patch("tradingagents.dataflows.net.requests.get",
                    side_effect=lambda *a, **k: called.append(a)), \
         pytest.raises(AnalysisProcessHasNoVendorKeys) as caught:
        fred._request("series", {"series_id": "DGS10"})
    assert _KEY not in str(caught.value)
    assert called == []
