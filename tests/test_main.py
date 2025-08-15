import pytest
import json
import sys
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from io import BytesIO

# Import the app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app, asset_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_store():
    """Clean asset store before each test"""
    asset_store.clear()
    yield
    asset_store.clear()


def test_upload_assets():
    """Test successful asset upload"""
    # Simple valid GeoJSON
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [173.596093, -41.216070],  #  NZ
        },
        "properties": {"address": "41 Opouri Road, Rai Valley 7194, New Zealand"},
    }

    response = client.post(
        "/assets/upload",
        data={"companyId": "test_company"},
        files={
            "assetFile": (
                "test.geojson",
                BytesIO(json.dumps(geojson).encode()),
                "application/json",
            )
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Assets uploaded successfully"
    assert data["processed_count"] == 1


@patch("main.sample_raster_value", return_value=42.5)
def test_get_assets_with_raster_sampling(mock_sample):
    """Test asset retrieval with raster sampling"""
    # Setup test data
    asset_store["test_company"] = [
        {
            "id": "test-id",
            "companyId": "test_company",
            "address": "123 Test Street",
            "latitude": -36.8485,
            "longitude": 174.7633,
        }
    ]

    response = client.get("/assets?companyId=test_company&lat=-36.8485&lon=174.7633")

    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 1
    assert assets[0]["raster_value"] == 42.5


@patch("main.sample_raster_value", return_value=9999)
def test_get_assets_out_of_bounds(mock_sample):
    """Test coordinates outside raster bounds"""
    asset_store["test_company"] = [
        {
            "id": "test-id",
            "companyId": "test_company",
            "address": "123 Test Street",
            "latitude": 0.0,
            "longitude": 0.0,
        }
    ]

    response = client.get("/assets?companyId=test_company&lat=0.0&lon=0.0")

    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 1
    assert assets[0]["raster_value"] == 9999


def test_get_assets_company_only():
    """Test retrieving all assets for a company"""
    asset_store["test_company"] = [
        {
            "id": "1",
            "companyId": "test_company",
            "address": "Street 1",
            "latitude": -36.8,
            "longitude": 174.7,
        },
        {
            "id": "2",
            "companyId": "test_company",
            "address": "Street 2",
            "latitude": -41.3,
            "longitude": 174.8,
        },
    ]

    response = client.get("/assets?companyId=test_company")

    assert response.status_code == 200
    assets = response.json()
    assert len(assets) == 2
    assert "raster_value" not in assets[0]


def test_upload_invalid_file():
    """Test invalid file upload error"""
    response = client.post(
        "/assets/upload",
        data={"companyId": "test_company"},
        files={
            "assetFile": ("test.geojson", BytesIO(b"invalid json"), "application/json")
        },
    )

    assert response.status_code == 400


def test_invalid_coordinates():
    """Test invalid coordinate validation"""
    # First, add a company so coordinate validation will trigger
    asset_store["test_company"] = [
        {
            "id": "test-id",
            "companyId": "test_company",
            "address": "123 Test Street",
            "latitude": -36.8485,
            "longitude": 174.7633,
        }
    ]

    # Test invalid latitude (should return 400)
    response = client.get("/assets?companyId=test_company&lat=100&lon=174")
    print(f"Status: {response.status_code}, Response: {response.json()}")

    assert response.status_code == 400
    response_data = response.json()

    # Check different possible response formats
    if "error" in response_data:
        error_message = response_data["error"]
    elif "detail" in response_data:
        detail = response_data["detail"]
        if isinstance(detail, dict) and "error" in detail:
            error_message = detail["error"]
        else:
            error_message = str(detail)
    else:
        error_message = str(response_data)

    assert "lat must be between -90 and 90" in error_message
