import ee
import os
import geemap
import datetime
import time

# Khởi tạo Earth Engine API
ee.Initialize(project='ee-bonglantrungmuoi')

# Thư mục dự án
project_dir = "D:/HaiDang/Biomass/GiaLai-Biomass"
data_dir = os.path.join(project_dir, "Data")
os.makedirs(data_dir, exist_ok=True)

print("="*80)
print("SCRIPT EXPORT DỮ LIỆU TỪ GEE CHO DỰ ÁN SINH KHỐI GIA LAI")
print("="*80)

# Tạo tên thư mục dựa trên thời gian
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"GiaLai_Biomass_{current_time}"
print(f"Dữ liệu sẽ được export vào Google Drive trong thư mục: {folder_name}")

# Khoảng thời gian nghiên cứu
start_date = ee.Date.fromYMD(2022, 10, 10)
end_date = ee.Date.fromYMD(2023, 10, 10)

# 1. Sử dụng shapefile Gia Lai có sẵn
print("\n[1/4] Đang đọc shapefile Gia Lai...")
# Sử dụng ranh giới từ Earth Engine
geometry = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/gia_lai")
print("  Đã tải ranh giới Gia Lai từ Earth Engine")

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

# Export dữ liệu Sentinel-2 vào Google Drive
print("  Đang chuẩn bị export dữ liệu Sentinel-2...")
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'ndvi', 'evi', 'mndwi', 'ndbi']
scale = 100  # Sử dụng độ phân giải thấp hơn để giảm kích thước

for band in bands:
    print(f"    Đang export band {band}...")
    band_data = s2Composite.select(band)
    
    # Export sang Google Drive
    task = ee.batch.Export.image.toDrive(
        image=band_data.clip(geometry),
        description=f'sentinel2_{band}',
        folder=folder_name,
        region=geometry.geometry().bounds(),
        scale=scale,
        maxPixels=1e9
    )
    task.start()
    print(f"    Đã bắt đầu export {band} vào Google Drive (theo dõi tại https://code.earthengine.google.com/tasks)")

# 3. Export dữ liệu DEM
print("\n[3/4] Đang xử lý dữ liệu DEM...")
glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30')
elevation = glo30.select('DEM').filterBounds(geometry).mosaic()
slope = ee.Terrain.slope(elevation)

# Export DEM và Slope vào Google Drive
print("  Đang export DEM...")
task_dem = ee.batch.Export.image.toDrive(
    image=elevation.clip(geometry),
    description='dem',
    folder=folder_name,
    region=geometry.geometry().bounds(),
    scale=100,
    maxPixels=1e9
)
task_dem.start()
print("  Đã bắt đầu export DEM vào Google Drive")

print("  Đang export Slope...")
task_slope = ee.batch.Export.image.toDrive(
    image=slope.clip(geometry),
    description='slope',
    folder=folder_name,
    region=geometry.geometry().bounds(),
    scale=100,
    maxPixels=1e9
)
task_slope.start()
print("  Đã bắt đầu export Slope vào Google Drive")

# 4. Export dữ liệu GEDI
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

# Export GEDI vào Google Drive
print("  Đang export GEDI...")
task_gedi = ee.batch.Export.image.toDrive(
    image=gediMosaic.clip(geometry),
    description='gedi_agbd',
    folder=folder_name,
    region=geometry.geometry().bounds(),
    scale=500,  # GEDI có độ phân giải thấp
    maxPixels=1e9
)
task_gedi.start()
print("  Đã bắt đầu export GEDI vào Google Drive")

print("\n=== HOÀN TẤT CHUẨN BỊ EXPORT TASKS ===")
print(f"Tất cả dữ liệu đang được export vào Google Drive, thư mục: {folder_name}")
print("Bạn có thể theo dõi tiến trình export tại: https://code.earthengine.google.com/tasks")
print("Sau khi tất cả các task hoàn thành, hãy tải dữ liệu từ Google Drive về thư mục Data")
print("Lưu ý: Quá trình export có thể mất 10-30 phút tùy thuộc vào kích thước dữ liệu và tải của server GEE")

# Kiểm tra trạng thái các task
print("\nĐang kiểm tra trạng thái các task export (sẽ kiểm tra trong 60 giây)...")
tasks = [task, task_dem, task_slope, task_gedi]
task_names = ['Sentinel-2', 'DEM', 'Slope', 'GEDI']

# Kiểm tra trong 60 giây
for i in range(6):
    print(f"\nLần kiểm tra {i+1}/6 (sau {i*10} giây):")
    for j, task in enumerate(tasks):
        status = task.status()
        state = status.get('state', 'UNKNOWN')
        print(f"  - {task_names[j]}: {state}")
    
    if i < 5:  # Không cần sleep ở lần cuối
        time.sleep(10)  # Chờ 10 giây

print("\nQuá trình kiểm tra kết thúc. Các task sẽ tiếp tục chạy trên máy chủ GEE.")
print("Vui lòng truy cập https://code.earthengine.google.com/tasks để xem trạng thái chi tiết.")
print("Sau khi tất cả task hoàn thành, hãy tải xuống dữ liệu từ Google Drive vào thư mục Data.")