from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Query
import json
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator
import logging
import rasterio
from rasterio.errors import RasterioIOError
import requests
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClimateX Challenge API",
    description="API for uploading and managing asset data from GeoJSON files",
    version="1.0.0",
)

# NZ habitat raster URL
NZ_HABITAT_RASTER_URL = "https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/nz_habitat_anticross_4326_1deg.tif"

# Cache for downloaded raster file
_raster_cache = {"file_path": None}

# In-memory storage for assets
# Structure: {companyId: [list of assets]}
asset_store: Dict[str, List[Dict]] = {}


class Asset(BaseModel):
    """Asset data model with validation"""

    id: str
    companyId: str
    address: str
    latitude: float
    longitude: float

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v


class UploadResponse(BaseModel):
    """Response model for successful uploads"""

    message: str
    processed_count: int


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    details: Optional[str] = None


def parse_geojson_file(file_content: str) -> List[Dict]:
    """Parse GeoJSON file and extract asset data"""
    try:
        geojson_data = json.loads(file_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

    if not isinstance(geojson_data, dict):
        raise ValueError("GeoJSON must be a JSON object")

    # Handle both FeatureCollection and single Feature
    features = []
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
    elif geojson_data.get("type") == "Feature":
        features = [geojson_data]
    else:
        raise ValueError("GeoJSON must be a Feature or FeatureCollection")

    if not features:
        raise ValueError("No features found in GeoJSON")

    assets = []
    for i, feature in enumerate(features):
        try:
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise ValueError(f"Feature {i + 1}: Invalid feature format")

            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})

            if geometry.get("type") != "Point":
                raise ValueError(f"Feature {i + 1}: Only Point geometry is supported")

            coordinates = geometry.get("coordinates", [])
            if len(coordinates) != 2:
                raise ValueError(
                    f"Feature {i + 1}: Point coordinates must have exactly 2 values [longitude, latitude]"
                )

            longitude, latitude = coordinates

            address = properties.get("address")
            if not address or not isinstance(address, str) or not address.strip():
                raise ValueError(
                    f"Feature {i + 1}: 'address' property is required and cannot be empty"
                )

            asset = {
                "address": address.strip(),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }

            for key, value in properties.items():
                if key != "address" and value is not None:
                    asset[key] = value

            assets.append(asset)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Feature {i + 1}: Error processing feature - {str(e)}")

    return assets


def download_raster_file() -> str:
    """Download and cache the NZ habitat raster file"""
    if _raster_cache["file_path"] and os.path.exists(_raster_cache["file_path"]):
        return _raster_cache["file_path"]

    try:
        logger.info("Downloading NZ habitat raster file...")
        response = requests.get(NZ_HABITAT_RASTER_URL, timeout=30)
        response.raise_for_status()

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        temp_file.write(response.content)
        temp_file.close()

        _raster_cache["file_path"] = temp_file.name
        logger.info(f"Raster file cached at: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        logger.error(f"Failed to download raster file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to download raster file: {str(e)}"},
        )


@app.get("/assets")
async def get_assets(
    companyId: str = Query(..., description="Company identifier"),
    lat: Optional[float] = Query(
        None, description="Latitude for exact coordinate matching"
    ),
    lon: Optional[float] = Query(
        None, description="Longitude for exact coordinate matching"
    ),
    min_lat: Optional[float] = Query(
        None, description="Minimum latitude for bounding box filter"
    ),
    max_lat: Optional[float] = Query(
        None, description="Maximum latitude for bounding box filter"
    ),
    min_lon: Optional[float] = Query(
        None, description="Minimum longitude for bounding box filter"
    ),
    max_lon: Optional[float] = Query(
        None, description="Maximum longitude for bounding box filter"
    ),
):
    """
    Get assets for a company with optional coordinate filtering and raster sampling.

    Parameters:
    - companyId: Required company identifier
    - lat, lon: Optional exact coordinate matching (returns assets with raster values)
    - min_lat, max_lat, min_lon, max_lon: Optional bounding box filtering

    Returns:
    - List of assets, with raster_value included if lat/lon provided
    - Empty list if no assets found
    """

    logger.info(
        f"Retrieving assets for companyId: {companyId}, lat: {lat}, lon: {lon}, "
        f"min_lat: {min_lat}, max_lat: {max_lat}, min_lon: {min_lon}, max_lon: {max_lon}"
    )
    try:
        # Validate companyId
        if not companyId or not companyId.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "companyId query parameter is required"},
            )

        company_id = companyId.strip()

        if company_id not in asset_store:
            return []

        assets = asset_store[company_id].copy()

        # If lat/lon provided, validate coordinates and filter for exact matches
        if lat is not None and lon is not None:
            if not -90 <= lat <= 90:
                raise HTTPException(
                    status_code=400, detail={"error": "lat must be between -90 and 90"}
                )
            if not -180 <= lon <= 180:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "lon must be between -180 and 180"},
                )

            # Filter assets for exact coordinate matches
            matching_assets = []
            for asset in assets:
                if asset["latitude"] == lat and asset["longitude"] == lon:
                    raster_value = sample_raster_value(lon, lat)

                    asset_with_raster = asset.copy()
                    asset_with_raster["raster_value"] = raster_value
                    matching_assets.append(asset_with_raster)

            return matching_assets

        # If only lat provided (without lon) or vice versa, return error
        elif lat is not None or lon is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Both lat and lon must be provided together for coordinate filtering"
                },
            )

        # Apply bounding box filtering if provided
        if any(param is not None for param in [min_lat, max_lat, min_lon, max_lon]):
            # Validate bounding box parameters
            if min_lat is not None and not -90 <= min_lat <= 90:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "min_lat must be between -90 and 90"},
                )
            if max_lat is not None and not -90 <= max_lat <= 90:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "max_lat must be between -90 and 90"},
                )
            if min_lon is not None and not -180 <= min_lon <= 180:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "min_lon must be between -180 and 180"},
                )
            if max_lon is not None and not -180 <= max_lon <= 180:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "max_lon must be between -180 and 180"},
                )

            if min_lat is not None and max_lat is not None and min_lat > max_lat:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "min_lat must be less than or equal to max_lat"},
                )
            if min_lon is not None and max_lon is not None and min_lon > max_lon:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "min_lon must be less than or equal to max_lon"},
                )

            # Filter assets within bounding box
            filtered_assets = []
            for asset in assets:
                asset_lat = asset["latitude"]
                asset_lon = asset["longitude"]

                if min_lat is not None and asset_lat < min_lat:
                    continue
                if max_lat is not None and asset_lat > max_lat:
                    continue

                if min_lon is not None and asset_lon < min_lon:
                    continue
                if max_lon is not None and asset_lon > max_lon:
                    continue

                filtered_assets.append(asset)

            return filtered_assets

        return assets

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving assets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error occurred while retrieving assets"},
        )


def sample_raster_value(lon: float, lat: float) -> Optional[float]:
    """Sample value from Band 1 of the NZ habitat raster at given coordinates"""
    try:
        raster_path = download_raster_file()

        with rasterio.open(raster_path) as dataset:
            if not (
                dataset.bounds.left <= lon <= dataset.bounds.right
                and dataset.bounds.bottom <= lat <= dataset.bounds.top
            ):
                logger.warning(f"Coordinates ({lon}, {lat}) are outside raster bounds")
                return 9999  # Indicator for out-of-bounds

            # Sample the raster at the given coordinates
            sampled_values = list(dataset.sample([(lon, lat)]))

            if sampled_values:
                value = sampled_values[0][0]
                if dataset.nodata is not None and value == dataset.nodata:
                    return None

                return float(value)
            else:
                return None

    except RasterioIOError as e:
        logger.error(f"Rasterio error sampling coordinates ({lon}, {lat}): {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error sampling raster at ({lon}, {lat}): {str(e)}")
        return None


def validate_and_create_assets(raw_assets: List[Dict], company_id: str) -> List[Asset]:
    """Validate asset data and create Asset objects"""
    validated_assets = []

    for i, asset_data in enumerate(raw_assets):
        try:
            # Add unique ID and companyId
            asset_data["id"] = str(uuid.uuid4())
            asset_data["companyId"] = company_id

            asset = Asset(**asset_data)
            validated_assets.append(asset)

        except Exception as e:
            raise ValueError(f"Asset {i + 1}: Validation failed - {str(e)}")

    return validated_assets


@app.post(
    "/assets/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def upload_assets(
    companyId: str = Form(..., description="Company identifier"),
    assetFile: UploadFile = File(..., description="GeoJSON file containing asset data"),
):
    """
    Upload GeoJSON asset data for a company.

    Expected GeoJSON format:
    - Feature or FeatureCollection with Point geometries
    - Each feature must have 'address' in properties
    - Coordinates in [longitude, latitude] format

    """
    try:
        # Validate company ID
        if not companyId or not companyId.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "companyId is required and cannot be empty"},
            )

        company_id = companyId.strip()

        # Validate file type
        filename = assetFile.filename or ""
        content_type = assetFile.content_type or ""

        if not (
            filename.lower().endswith((".json", ".geojson"))
            or "json" in content_type.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail={"error": "File must be a GeoJSON file (.json or .geojson)"},
            )

        # Read file content
        file_content = await assetFile.read()

        if not file_content:
            raise HTTPException(
                status_code=400, detail={"error": "Uploaded file is empty"}
            )

        try:
            file_text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400, detail={"error": "File must be UTF-8 encoded text"}
            )

        try:
            raw_assets = parse_geojson_file(file_text)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail={"error": f"GeoJSON parsing error: {str(e)}"}
            )

        if not raw_assets:
            raise HTTPException(
                status_code=400,
                detail={"error": "No valid assets found in GeoJSON file"},
            )

        # Validate and create Asset objects
        try:
            validated_assets = validate_and_create_assets(raw_assets, company_id)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail={"error": f"Validation error: {str(e)}"}
            )

        if company_id not in asset_store:
            asset_store[company_id] = []

        # Convert Asset objects to dictionaries for storage
        asset_dicts = [asset.model_dump() for asset in validated_assets]
        asset_store[company_id].extend(asset_dicts)

        logger.info(
            f"Successfully uploaded {len(validated_assets)} assets for company {company_id}"
        )

        return UploadResponse(
            message="Assets uploaded successfully",
            processed_count=len(validated_assets),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during asset upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error occurred during upload"},
        )


@app.get("/assets/{company_id}")
async def get_assets(company_id: str):
    """Get all assets for a company (for testing/debugging)"""
    if company_id not in asset_store:
        raise HTTPException(status_code=404, detail="Company not found")

    return {
        "company_id": company_id,
        "asset_count": len(asset_store[company_id]),
        "assets": asset_store[company_id],
    }


@app.get("/companies")
async def get_companies():
    """Get all companies with asset counts (for testing/debugging)"""
    return {
        company_id: {"asset_count": len(assets)}
        for company_id, assets in asset_store.items()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
