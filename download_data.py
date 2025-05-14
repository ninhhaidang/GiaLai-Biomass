import ee
import os
import geemap
import rasterio
from rasterio.transform import from_origin
import numpy as np
import geopandas as gpd
from datetime import datetime

# Khởi tạo Earth Engine API
ee.Initialize()

# Thư mục dự án
project_dir = "D:/HaiDang/Biomass/GiaLai-Biomass"  # Đường dẫn thực tế của người dùng
data_dir = os.path.join(project_dir, "Data")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir, "sentinel"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "dem"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "gedi"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "vector"), exist_ok=True)

print("="*80)
print("SCRIPT TẢI DỮ LIỆU TỪ GEE CHO DỰ ÁN SINH KHỐI GIA LAI")
print("="*80)

# Khoảng thời gian nghiên cứu
start_date = ee.Date.fromYMD(2022, 10, 10)
end_date = ee.Date.fromYMD(2023, 10, 10)

# 1. Sử dụng shapefile Gia Lai có sẵn
print("\n[1/4] Đang đọc shapefile Gia Lai...")
# Đường dẫn đến shapefile (điều chỉnh nếu cần)
shapefile_path = os.path.join(project_dir, "shapefile", "gia_lai.shp")
# Kiểm tra nếu đường dẫn không đúng, thử các vị trí khác
if not os.path.exists(shapefile_path):
    potential_paths = [
        os.path.join(project_dir, "gia_lai.shp"),
        os.path.join(project_dir, "vector", "gia_lai.shp"),
        os.path.join(project_dir, "boundary", "gia_lai.shp"),
        os.path.join(project_dir, "shapefile", "gia_lai.shp")
    ]
    for path in potential_paths:
        if os.path.exists(path):
            shapefile_path = path
            break

# Đọc shapefile
try:
    gdf = gpd.read_file(shapefile_path)
    print(f"  Đã đọc shapefile từ: {shapefile_path}")
    
    # Chuyển đổi geodataframe thành geometry cho Earth Engine
    geometry = geemap.gdf_to_ee(gdf)
    
    # Lấy bbox cho khu vực (sẽ dùng cho tải dữ liệu)
    bbox = geometry.geometry().bounds().getInfo()['coordinates'][0]
    min_lon = min(point[0] for point in bbox)
    max_lon = max(point[0] for point in bbox)
    min_lat = min(point[1] for point in bbox)
    max_lat = max(point[1] for point in bbox)
    print(f"  Ranh giới khu vực: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
except Exception as e:
    print(f"  Lỗi khi đọc shapefile: {str(e)}")
    print("  Đang sử dụng ranh giới Gia Lai từ GEE...")
    # Sử dụng ranh giới từ GEE nếu không đọc được shapefile
    geometry = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/gia_lai")
    bbox = geometry.geometry().bounds().getInfo()['coordinates'][0]
    min_lon = min(point[0] for point in bbox)
    max_lon = max(point[0] for point in bbox)
    min_lat = min(point[1] for point in bbox)
    max_lat = max(point[1] for point in bbox)

# 2. Tải dữ liệu Sentinel-2
print("\n[2/4] Đang xử lý dữ liệu Sentinel-2...")
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
filteredS2 = s2.filterBounds(geometry).filterDate(start_date, end_date)

# Sử dụng mặt nạ mây từ Cloud Score+
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
csPlusBands = csPlus.first().bandNames()
filteredS2WithCs = filteredS2.linkCollection(csPlus, csPlusBands)

# Hàm loại bỏ các pixel có điểm chất lượng thấp
def maskLowQA(image):
    mask = image.select('cs').gte(0.5)
    return image.updateMask(mask)

# Hàm áp dụng hệ số tỷ lệ
def scaleBands(image):
    return image.multiply(0.0001).copyProperties(image, ['system:time_start'])

# Tính các chỉ số thực vật
def addIndices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('mndwi')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('ndbi')
    evi = image.expression(
        '2.5 * ((NIR - RED)/(NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }).rename('evi')
    
    return image.addBands([ndvi, mndwi, ndbi, evi])

# Xử lý dữ liệu Sentinel-2
s2Processed = filteredS2WithCs.map(maskLowQA).select('B.*').map(scaleBands).map(addIndices)
s2Composite = s2Processed.median()

# Tải dữ liệu Sentinel-2
print("  Đang tải dữ liệu Sentinel-2...")
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'ndvi', 'evi', 'mndwi', 'ndbi']
scale = 10  # Độ phân giải 10m

for band in bands:
    print(f"    Đang tải band {band}...")
    band_data = s2Composite.select(band).clip(geometry)
    
    # Lấy dữ liệu dưới dạng mảng NumPy
    band_array = geemap.ee_to_numpy(
        band_data, 
        region=geometry.geometry(), 
        scale=scale
    )
    
    # Lưu dưới dạng GeoTIFF
    output_file = os.path.join(data_dir, "sentinel", f"sentinel2_{band}.tif")
    
    # Tạo GeoTIFF
    transform = from_origin(min_lon, max_lat, scale/111325, scale/111325)  # xres, yres in degrees
    
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=band_array.shape[0],
        width=band_array.shape[1],
        count=1,
        dtype=band_array.dtype,
        crs='+proj=longlat +datum=WGS84 +no_defs',
        transform=transform,
    ) as dst:
        dst.write(band_array, 1)
    
    print(f"      Đã lưu tại: {output_file}")

# 3. Tải dữ liệu DEM
print("\n[3/4] Đang xử lý dữ liệu DEM...")
glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30')
elevation = glo30.select('DEM').filterBounds(geometry).mosaic()
slope = ee.Terrain.slope(elevation)

# Tải DEM và Slope
dem_array = geemap.ee_to_numpy(
    elevation.clip(geometry), 
    region=geometry.geometry(), 
    scale=30
)

slope_array = geemap.ee_to_numpy(
    slope.clip(geometry), 
    region=geometry.geometry(), 
    scale=30
)

# Lưu DEM
output_dem = os.path.join(data_dir, "dem", "dem.tif")
transform_dem = from_origin(min_lon, max_lat, 30/111325, 30/111325)  # 30m resolution

with rasterio.open(
    output_dem,
    'w',
    driver='GTiff',
    height=dem_array.shape[0],
    width=dem_array.shape[1],
    count=1,
    dtype=dem_array.dtype,
    crs='+proj=longlat +datum=WGS84 +no_defs',
    transform=transform_dem,
) as dst:
    dst.write(dem_array, 1)

print(f"  Đã lưu DEM tại: {output_dem}")

# Lưu Slope
output_slope = os.path.join(data_dir, "dem", "slope.tif")

with rasterio.open(
    output_slope,
    'w',
    driver='GTiff',
    height=slope_array.shape[0],
    width=slope_array.shape[1],
    count=1,
    dtype=slope_array.dtype,
    crs='+proj=longlat +datum=WGS84 +no_defs',
    transform=transform_dem,
) as dst:
    dst.write(slope_array, 1)

print(f"  Đã lưu Slope tại: {output_slope}")

# 4. Tải dữ liệu GEDI
print("\n[4/4] Đang xử lý dữ liệu GEDI...")
gedi = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
gediFiltered = gedi.filter(ee.Filter.date(start_date, end_date)).filter(ee.Filter.bounds(geometry))

# Hàm tạo mặt nạ chất lượng cho GEDI
def qualityMask(image):
    return image.updateMask(image.select('l4_quality_flag').eq(1)) \
                .updateMask(image.select('degrade_flag').eq(0))

# Hàm tạo mặt nạ sai số cho GEDI
def errorMask(image):
    relative_se = image.select('agbd_se').divide(image.select('agbd'))
    return image.updateMask(relative_se.lte(0.3))

# Áp dụng các mặt nạ
gediProcessed = gediFiltered.map(qualityMask).map(errorMask)
gediMosaic = gediProcessed.mosaic().select('agbd')

# Tải GEDI
gedi_array = geemap.ee_to_numpy(
    gediMosaic.clip(geometry),
    region=geometry.geometry(),
    scale=500  # GEDI có độ phân giải thấp hơn
)

# Lưu GEDI
output_gedi = os.path.join(data_dir, "gedi", "gedi_agbd.tif")
transform_gedi = from_origin(min_lon, max_lat, 500/111325, 500/111325)  # 500m resolution

with rasterio.open(
    output_gedi,
    'w',
    driver='GTiff',
    height=gedi_array.shape[0],
    width=gedi_array.shape[1],
    count=1,
    dtype=gedi_array.dtype,
    crs='+proj=longlat +datum=WGS84 +no_defs',
    transform=transform_gedi,
) as dst:
    dst.write(gedi_array, 1)

print(f"  Đã lưu GEDI tại: {output_gedi}")

print("\n=== HOÀN TẤT TẢI DỮ LIỆU ===")
print(f"Tất cả dữ liệu đã được lưu tại: {data_dir}")
print("Dữ liệu đã sẵn sàng để sử dụng với script local.py")