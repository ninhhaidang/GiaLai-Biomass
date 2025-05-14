import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Khởi tạo Earth Engine API
ee.Initialize()

# Tạo bản đồ
map_gia_lai = geemap.Map()

print("Bắt đầu phân tích sinh khối Gia Lai...")

# Chọn vùng Gia Lai
geometry = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/gia_lai")
map_gia_lai.centerObject(geometry)

# Khoảng thời gian
startDate = ee.Date.fromYMD(2022, 10, 10)
endDate = ee.Date.fromYMD(2023, 10, 10)

# --- PHẦN 1: XỬ LÝ SENTINEL-2 ---
print("Đang xử lý dữ liệu Sentinel-2...")
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
filteredS2 = s2.filterBounds(geometry).filterDate(startDate, endDate)
s2Projection = ee.Image(filteredS2.first()).select('B4').projection()

# Xử lý mây
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
csPlusBands = csPlus.first().bandNames()
filteredS2WithCs = filteredS2.linkCollection(csPlus, csPlusBands)

def maskLowQA(image):
    mask = image.select('cs').gte(0.5)
    return image.updateMask(mask)

def scaleBands(image):
    return image.multiply(0.0001).copyProperties(image, ['system:time_start'])

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
    
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'L': 0.5
    }).rename('savi')
    
    return image.addBands([ndvi, mndwi, ndbi, evi, savi])

s2Processed = filteredS2WithCs.map(maskLowQA).select('B.*').map(scaleBands).map(addIndices)
s2Composite = s2Processed.median().setDefaultProjection(s2Projection).clip(geometry)

# --- PHẦN 2: XỬ LÝ DEM ---
print("Đang xử lý dữ liệu địa hình...")
glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30')
glo30Filtered = glo30.filter(ee.Filter.bounds(geometry)).select('DEM')
demProj = glo30Filtered.first().select(0).projection()
elevation = glo30Filtered.mosaic().rename('dem').setDefaultProjection(demProj).clip(geometry)
slope = ee.Terrain.slope(elevation).rename('slope').setDefaultProjection(demProj).clip(geometry)
demBands = elevation.addBands(slope)

# --- PHẦN 3: XỬ LÝ GEDI ---
print("Đang xử lý dữ liệu GEDI...")
gedi = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")

def qualityMask(image):
    return image.updateMask(image.select('l4_quality_flag').eq(1)) \
                .updateMask(image.select('degrade_flag').eq(0))

def errorMask(image):
    relative_se = image.select('agbd_se').divide(image.select('agbd'))
    return image.updateMask(relative_se.lte(0.3))

def slopeMask(image):
    return image.updateMask(slope.lt(30))

gediFiltered = gedi.filter(ee.Filter.date(startDate, endDate)).filter(ee.Filter.bounds(geometry))
gediProjection = ee.Image(gediFiltered.first()).select('agbd').projection()
gediProcessed = gediFiltered.map(qualityMask).map(errorMask).map(slopeMask)
gediMosaic = gediProcessed.mosaic().select('agbd').setDefaultProjection(gediProjection).clip(geometry)

# --- PHẦN 4: HUẤN LUYỆN MÔ HÌNH ---
print("Đang tích hợp dữ liệu và huấn luyện mô hình...")
gridScale = 100
gridProjection = ee.Projection('EPSG:3857').atScale(gridScale)

# Kết hợp các lớp dữ liệu
stacked = s2Composite.addBands(demBands).addBands(gediMosaic)
stacked = stacked.resample('bilinear')

stackedResampled = stacked.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=1024
).reproject(
    crs=gridProjection
).clip(geometry)

stackedResampled = stackedResampled.updateMask(stackedResampled.mask().gt(0))

# Danh sách các đặc trưng
predictors = s2Composite.bandNames().cat(demBands.bandNames())
predicted = 'agbd'  # Biến mục tiêu

# Lấy mẫu huấn luyện
classMask = stackedResampled.select([predicted]).mask().toInt().rename('class')
numSamples = 1000
training = stackedResampled.addBands(classMask).stratifiedSample(
    numPoints=numSamples,
    classBand='class',
    region=geometry,
    scale=gridScale,
    dropNulls=True,
    tileScale=16
)

# Huấn luyện Random Forest
model = ee.Classifier.smileRandomForest(50) \
                    .setOutputMode('REGRESSION') \
                    .train(
                        features=training,
                        classProperty=predicted,
                        inputProperties=predictors
                    )

# Đánh giá mô hình
predicted_samples = training.classify(
    classifier=model,
    outputName='agbd_predicted'
)

# Tính RMSE
def calculateRmse(input_samples):
    observed = ee.Array(input_samples.aggregate_array('agbd'))
    predicted = ee.Array(input_samples.aggregate_array('agbd_predicted'))
    rmse = observed.subtract(predicted).pow(2).reduce('mean', [0]).sqrt().get([0])
    return rmse

try:
    rmse = calculateRmse(predicted_samples)
    print('RMSE:', rmse.getInfo())
except Exception as e:
    print("Lỗi khi tính RMSE:", str(e))
    print("Tiếp tục với phân tích...")

# Dự đoán sinh khối
predictedImage = stackedResampled.classify(
    classifier=model,
    outputName='agbd'
)

# --- PHẦN 5: TÍNH TỔNG SINH KHỐI ---
print("Đang tính tổng sinh khối...")
worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
worldcoverResampled = worldcover.reduceResolution(
    reducer=ee.Reducer.mode(),
    maxPixels=1024
).reproject(
    crs=gridProjection
)

# Tạo mặt nạ lớp phủ (chỉ giữ các lớp rừng)
landCoverMask = worldcoverResampled.eq(10) \
    .Or(worldcoverResampled.eq(20)) \
    .Or(worldcoverResampled.eq(30)) \
    .Or(worldcoverResampled.eq(40)) \
    .Or(worldcoverResampled.eq(95))

# Áp dụng mặt nạ
predictedImageMasked = predictedImage.updateMask(landCoverMask)

# Tính diện tích pixel theo hecta
pixelAreaHa = ee.Image.pixelArea().divide(10000)

# Tính tổng sinh khối
predictedAgb = predictedImageMasked.multiply(pixelAreaHa)

try:
    # Tính toán thống kê
    stats = predictedAgb.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=30,
        maxPixels=1e10,
        tileScale=16
    )
    
    # Lấy tổng AGB
    totalAgb = stats.getNumber('agbd')
    print('Tổng sinh khối trên mặt đất (AGB) tại Gia Lai:', totalAgb.getInfo(), 'Mg')
except Exception as e:
    print("Lỗi khi tính tổng sinh khối:", str(e))

# --- HIỂN THỊ KẾT QUẢ ---
print("Đang hiển thị kết quả...")
# Hiển thị các lớp
map_gia_lai.addLayer(s2Composite, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'Sentinel-2 RGB')
map_gia_lai.addLayer(elevation, {'min': 0, 'max': 3000, 'palette': ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff']}, 'Độ cao')
map_gia_lai.addLayer(slope, {'min': 0, 'max': 60, 'palette': ['white', 'gray', 'black']}, 'Độ dốc')
map_gia_lai.addLayer(gediMosaic, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'red']}, 'GEDI Biomass')
map_gia_lai.addLayer(predictedImage, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'red']}, 'Predicted Biomass')

# --- PHẦN 6: LƯU TRỮ KẾT QUẢ ---
print("Đang lưu trữ kết quả...")

# Tạo thư mục kết quả
import os
output_dir = 'ket_qua_gia_lai'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Bản đồ tương tác HTML
map_gia_lai.to_html(f'{output_dir}/ban_do_sinh_khoi_gia_lai.html')
print(f"Đã lưu bản đồ tương tác tại: {output_dir}/ban_do_sinh_khoi_gia_lai.html")

# 2. Xuất bản đồ sinh khối dạng GeoTIFF về Google Drive
task = ee.batch.Export.image.toDrive(
    image=predictedImage,
    description='Sinh_khoi_Gia_Lai',
    folder='Ket_qua_GEE',
    scale=30,
    region=geometry,
    fileFormat='GeoTIFF'
)
task.start()
print("Đang xuất bản đồ sinh khối về Google Drive trong thư mục 'Ket_qua_GEE'")

# 3. Lưu kết quả số vào file CSV
try:
    results = {
        'RMSE': [rmse.getInfo()],
        'Tong_Sinh_Khoi_Mg': [totalAgb.getInfo()]
    }
    pd.DataFrame(results).to_csv(f'{output_dir}/ket_qua_sinh_khoi.csv')
    print(f"Đã lưu kết quả số liệu tại: {output_dir}/ket_qua_sinh_khoi.csv")
except Exception as e:
    print(f"Lỗi khi lưu kết quả số liệu: {str(e)}")

# 4. Tạo biểu đồ đánh giá mô hình
try:
    observed = predicted_samples.aggregate_array('agbd').getInfo()
    predicted = predicted_samples.aggregate_array('agbd_predicted').getInfo()

    plt.figure(figsize=(10, 10))
    plt.scatter(observed, predicted, alpha=0.5)
    plt.plot([0, max(observed)], [0, max(observed)], 'r--')
    plt.xlabel('Sinh khối quan sát (Mg/ha)')
    plt.ylabel('Sinh khối dự đoán (Mg/ha)')
    plt.title('So sánh giá trị sinh khối quan sát và dự đoán')
    plt.savefig(f'{output_dir}/do_chinh_xac_mo_hinh.png', dpi=300)
    print(f"Đã lưu biểu đồ đánh giá mô hình tại: {output_dir}/do_chinh_xac_mo_hinh.png")
except Exception as e:
    print(f"Lỗi khi tạo biểu đồ: {str(e)}")

print("\nQuá trình phân tích hoàn tất. Tất cả kết quả đã được lưu tại thư mục:", output_dir)
print("Bản đồ GeoTIFF đang được xuất về Google Drive, vui lòng kiểm tra sau vài phút.")

# Hiển thị bản đồ
map_gia_lai