import os
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import dask.array as da
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cupy as cp  # NumPy API trên GPU

# Thiết lập xử lý song song
cluster = LocalCUDACluster()  # Tận dụng GPU A4000
client = Client(cluster)
print(f"Dashboard: {client.dashboard_link}")

# Đường dẫn đến dữ liệu
data_dir = "D:/GiaLai_Project/Data"  # Thay đổi theo thư mục của bạn
output_dir = "D:/GiaLai_Project/Results"
os.makedirs(output_dir, exist_ok=True)

# Khoảng thời gian nghiên cứu
start_date = "2022-10-10"
end_date = "2023-10-10"
print("="*80)
print(f"THÔNG TIN DỰ ÁN: Phân tích sinh khối rừng Gia Lai")
print(f"Khoảng thời gian nghiên cứu: {start_date} đến {end_date}")
print(f"LƯU Ý: Đảm bảo dữ liệu Sentinel-2 và GEDI trong thư mục {data_dir} thuộc khoảng thời gian này")
print("="*80)

# 1. Xử lý dữ liệu Sentinel-2 với Dask để xử lý song song
def process_sentinel():
    print("Đang xử lý dữ liệu Sentinel-2...")
    sentinel_path = os.path.join(data_dir, "sentinel")
    
    # Lấy danh sách file
    sentinel_files = [os.path.join(sentinel_path, f) 
                     for f in os.listdir(sentinel_path) 
                     if f.endswith('.tif') or f.endswith('.jp2')]
    
    # Đọc tệp shapefile Gia Lai
    gialai = gpd.read_file(os.path.join(data_dir, "vector/gialai.shp"))
    
    # Xử lý từng band
    sentinel_data = {}
    meta = None
    
    for file in sentinel_files:
        band_name = os.path.basename(file).split('_')[2]  # Ví dụ: B04, B08...
        print(f"  Đang đọc band {band_name}...")
        
        with rasterio.open(file) as src:
            if meta is None:
                meta = src.meta.copy()
            
            # Cắt theo ranh giới Gia Lai
            out_image, out_transform = mask(src, gialai.geometry, crop=True)
            
            # Chuyển sang dask array để xử lý song song
            data_array = da.from_array(out_image[0], chunks=(2048, 2048))
            # Scale để chuyển thành độ phản xạ
            data_array = data_array * 0.0001
            
            sentinel_data[band_name] = data_array
            
    # Tính các chỉ số thực vật
    print("  Đang tính các chỉ số thực vật...")
    
    # Dùng dask để tính toán hiệu quả
    ndvi = (sentinel_data['B08'] - sentinel_data['B04']) / (sentinel_data['B08'] + sentinel_data['B04'] + 1e-8)
    
    evi = 2.5 * ((sentinel_data['B08'] - sentinel_data['B04']) / 
                 (sentinel_data['B08'] + 6 * sentinel_data['B04'] - 7.5 * sentinel_data['B02'] + 1))
    
    # Thêm các chỉ số vào kết quả
    sentinel_data['ndvi'] = ndvi
    sentinel_data['evi'] = evi
    
    # Chuyển thành numpy arrays (tính toán thực)
    for band in sentinel_data:
        sentinel_data[band] = sentinel_data[band].compute()
    
    return sentinel_data, meta, out_transform

# 2. Xử lý DEM với CuPy để tận dụng GPU
def process_dem(gialai):
    print("Đang xử lý dữ liệu DEM...")
    dem_file = os.path.join(data_dir, "dem/glo30.tif")
    
    with rasterio.open(dem_file) as src:
        dem, dem_transform = mask(src, gialai.geometry, crop=True)
        dem = dem[0]
        dem_meta = src.meta.copy()
    
    # Chuyển lên GPU để tính toán nhanh hơn
    dem_gpu = cp.array(dem)
    
    # Tính độ dốc trên GPU
    dx, dy = cp.gradient(dem_gpu)
    slope_gpu = cp.sqrt(dx**2 + dy**2)
    
    # Chuyển về CPU
    slope = cp.asnumpy(slope_gpu)
    dem = cp.asnumpy(dem_gpu)
    
    return {'dem': dem, 'slope': slope}, dem_meta, dem_transform

# 3. Huấn luyện mô hình RandomForest
def train_model(sentinel_data, dem_data, gedi_data):
    print("Đang huấn luyện mô hình Random Forest...")
    
    # Kết hợp các đặc trưng
    features = {}
    for band, data in sentinel_data.items():
        features[band] = data
    
    for name, data in dem_data.items():
        features[name] = data
    
    # Lấy mẫu dữ liệu huấn luyện (không sử dụng tất cả pixel)
    # Chia nhỏ giảm áp lực lên RAM
    sample_size = 100000  # Số mẫu để huấn luyện
    
    X = []
    y = []
    valid_mask = ~np.isnan(gedi_data)
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) > sample_size:
        # Chọn ngẫu nhiên mẫu từ những điểm có dữ liệu
        sample_idx = np.random.choice(len(valid_indices[0]), sample_size, replace=False)
        rows = valid_indices[0][sample_idx]
        cols = valid_indices[1][sample_idx]
    else:
        rows = valid_indices[0]
        cols = valid_indices[1]
    
    for i, j in zip(rows, cols):
        feature_vector = [features[k][i, j] for k in features if not np.isnan(features[k][i, j])]
        if len(feature_vector) == len(features):  # Kiểm tra không có giá trị NaN
            X.append(feature_vector)
            y.append(gedi_data[i, j])
    
    # Sử dụng Random Forest với cài đặt tận dụng đa nhân của CPU
    rf = RandomForestRegressor(
        n_estimators=100,
        n_jobs=-1,  # Sử dụng tất cả CPU cores
        random_state=42
    )
    rf.fit(X, y)
    
    # Tính RMSE trên tập huấn luyện
    y_pred = rf.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"  RMSE trên tập huấn luyện: {rmse:.4f}")
    
    return rf, features

# Hàm chính
def main():
    # Đọc dữ liệu shapefile
    gialai = gpd.read_file(os.path.join(data_dir, "vector/gialai.shp"))
    
    # 1. Xử lý Sentinel-2
    sentinel_data, sentinel_meta, sentinel_transform = process_sentinel()
    
    # 2. Xử lý DEM
    dem_data, dem_meta, dem_transform = process_dem(gialai)
    
    # 3. Đọc dữ liệu GEDI (đã được tiền xử lý)
    print("Đang đọc dữ liệu GEDI...")
    with rasterio.open(os.path.join(data_dir, "gedi/gedi_agbd.tif")) as src:
        gedi_data, gedi_transform = mask(src, gialai.geometry, crop=True)
        gedi_data = gedi_data[0]
    
    # 4. Huấn luyện mô hình
    model, features = train_model(sentinel_data, dem_data, gedi_data)
    
    # 5. Dự đoán sinh khối
    print("Đang dự đoán sinh khối...")
    
    # Tạo mảng đặc trưng cho dự đoán
    rows, cols = dem_data['dem'].shape
    prediction_map = np.full((rows, cols), np.nan)
    
    # Dự đoán theo blocks để tiết kiệm bộ nhớ
    block_size = 1000
    for row_start in range(0, rows, block_size):
        for col_start in range(0, cols, block_size):
            row_end = min(row_start + block_size, rows)
            col_end = min(col_start + block_size, cols)
            
            # Lấy mẫu trong block
            block_features = []
            block_indices = []
            
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    # Kiểm tra điều kiện độ dốc và mặt nạ
                    if dem_data['slope'][i, j] <= 30:
                        feature_vector = [features[k][i, j] for k in features 
                                         if not np.isnan(features[k][i, j])]
                        if len(feature_vector) == len(features):
                            block_features.append(feature_vector)
                            block_indices.append((i, j))
            
            if block_features:
                # Dự đoán sinh khối cho block
                block_predictions = model.predict(block_features)
                
                # Gán kết quả vào bản đồ dự đoán
                for idx, (i, j) in enumerate(block_indices):
                    prediction_map[i, j] = block_predictions[idx]
    
    # 6. Lưu kết quả
    print("Đang lưu kết quả...")
    
    # Lưu bản đồ sinh khối dưới dạng GeoTIFF
    with rasterio.open(
        os.path.join(output_dir, "sinh_khoi_gia_lai.tif"),
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=prediction_map.dtype,
        crs=sentinel_meta['crs'],
        transform=sentinel_transform
    ) as dst:
        dst.write(prediction_map, 1)
    
    # Tính tổng sinh khối
    pixel_area_ha = 0.01  # Diện tích pixel theo hecta (giả định độ phân giải 10m)
    total_biomass = np.nansum(prediction_map) * pixel_area_ha
    print(f"Tổng sinh khối ước tính: {total_biomass:.2f} Mg")
    
    # Lưu kết quả số liệu 
    results = pd.DataFrame({
        'Metric': ['RMSE', 'Total_Biomass_Mg'],
        'Value': [np.sqrt(np.mean((model.predict(block_features) - block_predictions) ** 2)), total_biomass]
    })
    results.to_csv(os.path.join(output_dir, "ket_qua_sinh_khoi.csv"), index=False)
    
    # Tạo bản đồ
    plt.figure(figsize=(12, 10))
    plt.imshow(prediction_map, cmap='viridis')
    plt.colorbar(label='Sinh khối (Mg/ha)')
    plt.title('Bản đồ sinh khối rừng tỉnh Gia Lai')
    plt.savefig(os.path.join(output_dir, "ban_do_sinh_khoi.png"), dpi=300)
    
    print(f"Hoàn tất! Kết quả đã được lưu trong thư mục {output_dir}")

if __name__ == "__main__":
    main()