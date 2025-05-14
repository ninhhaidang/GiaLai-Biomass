Map.centerObject(geometry);

// Chọn khoảng thời gian: từ 10/10/2022 đến 10/10/2023
var startDate = ee.Date.fromYMD(2022, 10, 10);
var endDate = ee.Date.fromYMD(2023, 10, 10);

// Lọc ảnh Sentinel-2 theo thời gian và khu vực
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var filteredS2 = s2.filterBounds(geometry)
    .filterDate(startDate, endDate);

// Lấy hệ chiếu trước khi xử lý
var s2Projection = ee.Image(filteredS2.first()).select('B4').projection();

// Hàm áp dụng hệ số tỷ lệ để chuyển giá trị pixel thành độ phản xạ
var scaleBands = function (image) {
    return image.multiply(0.0001).copyProperties(image, ['system:time_start']);
};

// Sử dụng mặt nạ mây từ Cloud Score+
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csPlusBands = csPlus.first().bandNames();
var filteredS2WithCs = filteredS2.linkCollection(csPlus, csPlusBands);

// Hàm loại bỏ các pixel có điểm chất lượng thấp
var maskLowQA = function (image) {
    var mask = image.select('cs').gte(0.5);
    return image.updateMask(mask);
};

// Hàm tính toán các chỉ số quang học, bổ sung thêm các chỉ số nâng cao
var addIndices = function (image) {
    var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
    var mndwi = image.normalizedDifference(['B3', 'B11']).rename('mndwi');
    var ndbi = image.normalizedDifference(['B11', 'B8']).rename('ndbi');

    var evi = image.expression(
        '2.5 * ((NIR - RED)/(NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }).rename('evi');

    var bsi = image.expression(
        '((X + Y) - (A + B)) / ((X + Y) + (A + B))', {
        'X': image.select('B11'),
        'Y': image.select('B4'),
        'A': image.select('B8'),
        'B': image.select('B2'),
    }).rename('bsi');

    // Các chỉ số bổ sung
    var savi = image.expression(
        '((NIR - RED) / (NIR + RED + L)) * (1 + L)', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'L': 0.5
    }).rename('savi');

    var gci = image.expression(
        '(NIR / GREEN) - 1', {
        'NIR': image.select('B8'),
        'GREEN': image.select('B3')
    }).rename('gci');

    var arvi = image.expression(
        '(NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE))', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }).rename('arvi');

    var ndmi = image.normalizedDifference(['B8', 'B11']).rename('ndmi');

    var cire = image.expression(
        '(REDEDGE - RED) / RED', {
        'REDEDGE': image.select('B5'),
        'RED': image.select('B4')
    }).rename('cire');

    return image.addBands([ndvi, mndwi, ndbi, evi, bsi, savi, gci, arvi, ndmi, cire]);
};

// Áp dụng các bước tiền xử lý
var s2Processed = filteredS2WithCs
    .map(maskLowQA)
    .select('B.*')
    .map(scaleBands)
    .map(addIndices);  // Bổ sung các chỉ số mới vào quá trình xử lý

// Tạo ảnh composite Sentinel-2 và cắt theo geometry
var s2Composite = s2Processed.median()  // Sử dụng trung bình các ảnh để tạo composite
    .setDefaultProjection(s2Projection)  // Đảm bảo hệ chiếu đúng
    .clip(geometry);  // Cắt theo vùng nghiên cứu

// Hiển thị composite với các kênh quang học cơ bản (RGB)
Map.addLayer(s2Composite, { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3 }, 'Sentinel-2 Composite');

// ------------------------------
// Phần 2: Xử lý DEM và Tính độ dốc (Sử dụng GLO-30)
// ------------------------------

// Chọn dữ liệu DEM GLO-30
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');
var glo30Filtered = glo30.filter(ee.Filter.bounds(geometry))
    .select('DEM');

// Lấy hệ chiếu từ GLO-30
var demProj = glo30Filtered.first().select(0).projection();

// Tạo ảnh độ cao từ GLO-30 và cập nhật hệ chiếu
var elevation = glo30Filtered.mosaic().rename('dem')
    .setDefaultProjection(demProj) // Đảm bảo sử dụng hệ chiếu của GLO-30
    .clip(geometry);

// Tính toán độ dốc từ ảnh độ cao (DEM)
var slope = ee.Terrain.slope(elevation).rename('slope')
    .setDefaultProjection(demProj) // Đảm bảo sử dụng hệ chiếu của GLO-30
    .clip(geometry);

// Kết hợp ảnh độ cao và độ dốc thành một ảnh
var demBands = elevation.addBands(slope);

// Thiết lập các tham số hiển thị cho ảnh độ cao
var elevationVis = {
    min: 0,
    max: 3000,
    palette: ['0000ff', '00ffff', 'ffff00', 'ff0000', 'ffffff'],
};
Map.addLayer(elevation, elevationVis, 'Độ cao GLO-30');

// Thiết lập các tham số hiển thị cho ảnh độ dốc
var slopeVis = {
    min: 0,
    max: 60,
    palette: ['white', 'gray', 'black'],
};
Map.addLayer(slope, slopeVis, 'Độ dốc GLO-30');

// Căn chỉnh bản đồ theo vị trí của khu vực nghiên cứu
Map.centerObject(geometry, 10);

// ------------------------------
// Phần 3: Xử lý dữ liệu GEDI L4A
// ------------------------------

// Chọn dữ liệu GEDI L4A
var gedi = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY");

// Hàm tạo mặt nạ chất lượng cho GEDI
var qualityMask = function (image) {
    return image.updateMask(image.select('l4_quality_flag').eq(1))
        .updateMask(image.select('degrade_flag').eq(0));
};

// Hàm tạo mặt nạ sai số cho GEDI
var errorMask = function (image) {
    var relative_se = image.select('agbd_se')
        .divide(image.select('agbd'));
    return image.updateMask(relative_se.lte(0.3));
};

// Hàm tạo mặt nạ độ dốc cho GEDI
var slopeMask = function (image) {
    return image.updateMask(slope.lt(30)); // Sử dụng độ dốc tính từ phần 2
};

// Lọc dữ liệu GEDI theo thời gian và khu vực
var gediFiltered = gedi.filter(ee.Filter.date(startDate, endDate))
    .filter(ee.Filter.bounds(geometry));

// Lấy hệ chiếu của GEDI
var gediProjection = ee.Image(gediFiltered.first())
    .select('agbd').projection();

// Áp dụng các mặt nạ và xử lý dữ liệu GEDI
var gediProcessed = gediFiltered
    .map(qualityMask)
    .map(errorMask)
    .map(slopeMask);

// Ghép các ảnh GEDI lại với nhau (mosaic)
var gediMosaic = gediProcessed.mosaic()
    .select('agbd')
    .setDefaultProjection(gediProjection) // Đảm bảo hệ chiếu của GEDI
    .clip(geometry);

// Hiển thị kết quả của GEDI Biomass Density
Map.addLayer(gediMosaic, { min: 0, max: 100, palette: ['blue', 'green', 'yellow', 'red'] }, 'GEDI Biomass Density');

// ------------------------------
// Phần 4: Xuất dữ liệu thành Assets
// ------------------------------
var exportPath = 'users/huutruongnb/hihitest/';
Export.image.toAsset({
    image: s2Composite.clip(geometry),
    description: 'S2_Composite_Export',
    assetId: exportPath + 's2_composite',
    region: geometry,
    scale: 100,
    maxPixels: 1e10
});

Export.image.toAsset({
    image: demBands.clip(geometry),
    description: 'DEM_Bands_Export',
    assetId: exportPath + 'dem_bands',
    region: geometry,
    scale: 100,
    maxPixels: 1e10
});

Export.image.toAsset({
    image: gediMosaic.clip(geometry),
    description: 'GEDI_Mosaic_Export',
    assetId: exportPath + 'gedi_mosaic',
    region: geometry,
    scale: 100,
    maxPixels: 1e10
});

var exportPath = 'users/huutruongnb/hihitest/';
// ------------------------------
// Phần 5: Resampling & Huấn luyện mô hình hồi quy (Random Forest)
// ------------------------------

var s2Composite = ee.Image(exportPath + 's2_composite');
var demBands = ee.Image(exportPath + 'dem_bands');
var gediMosaic = ee.Image(exportPath + 'gedi_mosaic');
var gridScale = 100;
var gridProjection = ee.Projection('EPSG:3857').atScale(gridScale);

// Kết hợp các lớp dữ liệu: s2Composite, demBands, gediMosaic
var stacked = s2Composite.addBands(demBands).addBands(gediMosaic);

// Resample dữ liệu sử dụng bilinear
stacked = stacked.resample('bilinear');

// Giảm độ phân giải và tái chiếu
var stackedResampled = stacked.reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 1024
}).reproject({
    crs: gridProjection
}).clip(geometry);

// Áp dụng mask để loại bỏ các pixel không hợp lệ
stackedResampled = stackedResampled.updateMask(stackedResampled.mask().gt(0));

// Cập nhật danh sách các đặc trưng (predictors) bao gồm 
var predictors = s2Composite.bandNames().cat(demBands.bandNames());
var predicted = gediMosaic.bandNames().get(0); // Biến mục tiêu (AGBD)
print('Predictors', predictors);
print('Predicted', predicted);

// Chọn hình ảnh đặc trưng và mục tiêu
var predictorImage = stackedResampled.select(predictors);
var predictedImage = stackedResampled.select([predicted]);

// Tạo mask lớp để lấy mẫu phân tầng
var classMask = predictedImage.mask().toInt().rename('class');

// Lấy mẫu phân tầng để tạo tập huấn luyện
var numSamples = 1000;
var training = stackedResampled.addBands(classMask)
    .stratifiedSample({
        numPoints: numSamples,
        classBand: 'class',
        region: geometry,
        scale: gridScale,
        classValues: [0, 1],
        classPoints: [0, numSamples],
        dropNulls: true,
        tileScale: 16
    });
print('Số đặc trưng được trích xuất', training.size());
print('Đặc trưng huấn luyện mẫu', training.first());

// Huấn luyện mô hình Random Forest
var model = ee.Classifier.smileRandomForest(50)
    .setOutputMode('REGRESSION')
    .train({
        features: training,
        classProperty: predicted,
        inputProperties: predictors
    });

// Dự đoán trên tập huấn luyện để tính RMSE
var predicted = training.classify({
    classifier: model,
    outputName: 'agbd_predicted'
});

// Hàm tính RMSE
var calculateRmse = function (input) {
    var observed = ee.Array(input.aggregate_array('agbd'));
    var predicted = ee.Array(input.aggregate_array('agbd_predicted'));
    var rmse = observed.subtract(predicted).pow(2)
        .reduce('mean', [0]).sqrt().get([0]);
    return rmse;
};
var rmse = calculateRmse(predicted);
print('RMSE', rmse);

// Tạo biểu đồ phân tán
var chart = ui.Chart.feature.byFeature({
    features: predicted.select(['agbd', 'agbd_predicted']),
    xProperty: 'agbd',
    yProperties: ['agbd_predicted'],
}).setChartType('ScatterChart')
    .setOptions({
        title: 'Mật độ Biomass Trên mặt đất (Mg/Ha)',
        dataOpacity: 0.8,
        hAxis: { 'title': 'Quan sát' },
        vAxis: { 'title': 'Dự đoán' },
        legend: { position: 'right' },
        series: {
            0: {
                visibleInLegend: false,
                color: '#525252',
                pointSize: 3,
                pointShape: 'triangle',
            },
        },
        trendlines: {
            0: {
                type: 'linear',
                color: 'black',
                lineWidth: 1,
                pointSize: 0,
                labelInLegend: 'Đường phù hợp tuyến tính',
                visibleInLegend: true,
                showR2: true
            }
        },
        chartArea: { left: 100, bottom: 100, width: '50%' },
    });
print(chart);

// Dự đoán trên toàn bộ hình ảnh
var predictedImage = stackedResampled.classify({
    classifier: model,
    outputName: 'agbd'
});

// Xuất hình ảnh dự đoán
Export.image.toAsset({
    image: predictedImage.clip(geometry),
    description: 'Predicted_Image_Export',
    assetId: exportPath + 'predicted_agbd',
    region: geometry,
    scale: gridScale,
    maxPixels: 1e10
});

// ------------------------------
// Phần 6: Ước tính Tổng Biomass (AGB)
// ------------------------------
var s2Composite = ee.Image(exportPath + 's2_composite');
var predictedImage = ee.Image(exportPath + 'predicted_agbd');
var gridProjection = s2Composite.projection();
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first();
var worldcoverResampled = worldcover.reduceResolution({
    reducer: ee.Reducer.mode(),
    maxPixels: 1024
}).reproject({
    crs: gridProjection
});
var landCoverMask = worldcoverResampled.eq(10)
    .or(worldcoverResampled.eq(20))
    .or(worldcoverResampled.eq(30))
    .or(worldcoverResampled.eq(40))
    .or(worldcoverResampled.eq(95));
var predictedImageMasked = predictedImage.updateMask(landCoverMask);
var pixelAreaHa = ee.Image.pixelArea().divide(10000);
var predictedAgb = predictedImageMasked.multiply(pixelAreaHa);
var stats = predictedAgb.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geometry,
    scale: 30,
    maxPixels: 1e10,
    tileScale: 16
});
var totalAgb = stats.getNumber('agbd');
print('Tổng AGB (Mg)', totalAgb);