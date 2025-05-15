import ee
import os
import geemap
import rasterio
from rasterio.transform import from_origin
import numpy as np
import geopandas as gpd
from datetime import datetime
import time

# Khởi tạo Earth Engine API
ee.Initialize(project='ee-bonglantrungmuoi')
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

# Tải dữ liệu Sentinel-2 bằng cách chia nhỏ khu vực
print("  Đang tải dữ liệu Sentinel-2...")
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'ndvi', 'evi', 'mndwi', 'ndbi']
scale = 30  # Tăng độ phân giải lên 30m thay vì 10m để giảm kích thước

# Chia khu vực thành lưới nhỏ để tải
def download_in_grid(image, region, output_file, scale=30, grid_size=2):
    # Tính toán kích thước mỗi phần
    region_info = region.getInfo()
    coords = region_info['coordinates'][0]
    
    # Tính min/max lon/lat
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Chia thành lưới grid_size x grid_size
    lon_step = (max_lon - min_lon) / grid_size
    lat_step = (max_lat - min_lat) / grid_size
    
    # Tạo các mảng numpy để lưu kết quả
    arrays = []
    
    print(f"    Chia khu vực thành lưới {grid_size}x{grid_size}")
    
    # Xử lý từng phần nhỏ
    for i in range(grid_size):
        for j in range(grid_size):
            cell_min_lon = min_lon + i * lon_step
            cell_max_lon = min_lon + (i + 1) * lon_step
            cell_min_lat = min_lat + j * lat_step
            cell_max_lat = min_lat + (j + 1) * lat_step
            
            # Tạo geometry cho phần nhỏ
            cell_geom = ee.Geometry.Rectangle([cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat])
            
            print(f"    Đang tải phần {i*grid_size + j + 1}/{grid_size*grid_size}...")
            try:
                # Lấy dữ liệu từ phần nhỏ
                array = geemap.ee_to_numpy(
                    image.clip(cell_geom),
                    region=cell_geom,
                    scale=scale
                )
                
                arrays.append((array, i, j))
                time.sleep(1)  # Đợi 1 giây để tránh quá tải API
            except Exception as e:
                print(f"    Lỗi khi tải phần {i*grid_size + j + 1}: {str(e)}")
                # Nếu lỗi, thử tăng scale để giảm kích thước
                try:
                    print(f"    Thử lại với độ phân giải thấp hơn (scale={scale*2})...")
                    array = geemap.ee_to_numpy(
                        image.clip(cell_geom),
                        region=cell_geom,
                        scale=scale*2
                    )
                    arrays.append((array, i, j))
                    time.sleep(1)
                except Exception as e:
                    print(f"    Không thể tải phần {i*grid_size + j + 1}: {str(e)}")
    
    # Gộp các mảng lại với nhau
    if not arrays:
        raise Exception("Không thể tải dữ liệu từ bất kỳ phần nào của lưới")
    
    # Sắp xếp các mảng theo vị trí i, j
    arrays.sort(key=lambda x: (x[1], x[2]))
    
    # Gộp theo chiều ngang (cùng hàng)
    rows = []
    for i in range(grid_size):
        row_arrays = [arr for arr, row, col in arrays if row == i]
        if row_arrays:
            rows.append(np.hstack(row_arrays))
    
    # Gộp theo chiều dọc
    if rows:
        final_array = np.vstack(rows)
    else:
        raise Exception("Không thể gộp các mảnh dữ liệu")
    
    # Lưu dưới dạng GeoTIFF
    transform = from_origin(min_lon, max_lat, scale/111325, scale/111325)
    
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=final_array.shape[0],
        width=final_array.shape[1],
        count=1,
        dtype=final_array.dtype,
        crs='+proj=longlat +datum=WGS84 +no_defs',
        transform=transform,
    ) as dst:
        dst.write(final_array, 1)
    
    print(f"      Đã lưu tại: {output_file}")
    return final_array

# Tải từng band
for band in bands:
    print(f"    Đang tải band {band}...")
    band_data = s2Composite.select(band)
    
    output_file = os.path.join(data_dir, "sentinel", f"sentinel2_{band}.tif")
    
    try:
        # Thử tải toàn bộ với độ phân giải thấp hơn
        download_in_grid(
            image=band_data,
            region=geometry.geometry().bounds(),
            output_file=output_file,
            scale=scale,
            grid_size=3  # Chia thành lưới 3x3 = 9 phần nhỏ
        )
    except Exception as e:
        print(f"    Lỗi khi tải band {band}: {str(e)}")
        print(f"    Thử lại với độ phân giải thấp hơn...")
        try:
            download_in_grid(
                image=band_data,
                region=geometry.geometry().bounds(),
                output_file=output_file,
                scale=scale*2,  # Giảm độ phân giải xuống một nửa
                grid_size=4     # Chia thành lưới 4x4 = 16 phần nhỏ
            )
        except Exception as e:
            print(f"    Không thể tải band {band}: {str(e)}")

# 3. Tải dữ liệu DEM
print("\n[3/4] Đang xử lý dữ liệu DEM...")
glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30')
elevation = glo30.select('DEM').filterBounds(geometry).mosaic()
slope = ee.Terrain.slope(elevation)

# Tải DEM và Slope với cùng phương pháp chia lưới
print("  Đang tải dữ liệu DEM...")
output_dem = os.path.join(data_dir, "dem", "dem.tif")
try:
    download_in_grid(
        image=elevation,
        region=geometry.geometry().bounds(),
        output_file=output_dem,
        scale=90,  # DEM với độ phân giải 90m
        grid_size=2
    )
except Exception as e:
    print(f"  Lỗi khi tải DEM: {str(e)}")

print("  Đang tải dữ liệu Slope...")
output_slope = os.path.join(data_dir, "dem", "slope.tif")
try:
    download_in_grid(
        image=slope,
        region=geometry.geometry().bounds(),
        output_file=output_slope,
        scale=90,  # Slope với độ phân giải 90m
        grid_size=2
    )
except Exception as e:
    print(f"  Lỗi khi tải Slope: {str(e)}")

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

# Tải GEDI với độ phân giải thấp (phù hợp với dữ liệu GEDI)
print("  Đang tải dữ liệu GEDI...")
output_gedi = os.path.join(data_dir, "gedi", "gedi_agbd.tif")
try:
    download_in_grid(
        image=gediMosaic,
        region=geometry.geometry().bounds(),
        output_file=output_gedi,
        scale=500,  # GEDI có độ phân giải thấp hơn ~500m
        grid_size=1  # Không cần chia nhỏ vì GEDI đã có độ phân giải thấp
    )
except Exception as e:
    print(f"  Lỗi khi tải GEDI: {str(e)}")
    print("  Thử lại với độ phân giải thấp hơn...")
    try:
        download_in_grid(
            image=gediMosaic,
            region=geometry.geometry().bounds(),
            output_file=output_gedi,
            scale=1000,  # Giảm độ phân giải xuống 1km
            grid_size=1
        )
    except Exception as e:
        print(f"  Không thể tải GEDI: {str(e)}")

print("\n=== HOÀN TẤT TẢI DỮ LIỆU ===")
print(f"Tất cả dữ liệu đã được lưu tại: {data_dir}")
print("Dữ liệu đã sẵn sàng để sử dụng với script local.py")