import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MultipleLocator

import Testing
import Training

def haversine_distance(coords_true, coords_pred):
    R = 6371.0  
    lon1, lat1 = np.radians(coords_true[:, 0]), np.radians(coords_true[:, 1])
    lon2, lat2 = np.radians(coords_pred[:, 0]), np.radians(coords_pred[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

def load_csv_data(csv_path, window_size=4):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file '{csv_path}'!")

    # Thêm low_memory=False để tắt cảnh báo DtypeWarning
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    
    df['STORM_SPEED'] = pd.to_numeric(df['STORM_SPEED'], errors='coerce')
    df['STORM_DIR'] = pd.to_numeric(df['STORM_DIR'], errors='coerce')
    
    df['STORM_SPEED'] = df['STORM_SPEED'].ffill().fillna(0.0)
    df['STORM_DIR'] = df['STORM_DIR'].ffill().fillna(0.0)

    
    df = df.reset_index(drop=True)
    
    cyclones_data = []
    initial_coords = [] 

    for storm_id, group in df.groupby('ID', sort=False):
        
        # Lấy các cột gốc (Lúc này mảng đã sạch 100% số)
        lon = group['LON'].to_numpy(dtype=np.float64)
        lat = group['LAT'].to_numpy(dtype=np.float64)
        speed = group['STORM_SPEED'].to_numpy(dtype=np.float64)
        
        # Giải quyết điểm mù 360 độ bằng cách tách STORM_DIR thành Sin và Cos
        dir_rad = np.radians(group['STORM_DIR'].to_numpy(dtype=np.float64))
        sin_dir = np.sin(dir_rad)
        cos_dir = np.cos(dir_rad)
        
        coords = np.column_stack((lon, lat, speed, sin_dir, cos_dir))
        
        start_lon, start_lat = coords[0, 0], coords[0, 1] 

        # Xử lý bẫy vượt kinh tuyến 180 độ 
        lon_diff = coords[:, 0] - start_lon
        lon_diff = (lon_diff + 180) % 360 - 180

        # Lưu lại tọa độ tương đối
        coords[:, 0] = lon_diff
        coords[:, 1] = coords[:, 1] - start_lat

        if coords.ndim == 2 and coords.shape[0] >= 8:
            cyclones_data.append(coords)
            initial_coords.append((start_lon, start_lat))

    return cyclones_data, initial_coords

# 1. NẠP DỮ LIỆU
WINDOW_SIZE = 4
first_hid_cfg = [2, 25, 8]
second_hid_cfg = [2, 50, 10]

print('1. Đang nạp dữ liệu bão (Tích hợp Tọa độ, Vận tốc và Hướng di chuyển)...')
all_cyclones, all_initials = load_csv_data('ibtracs-tbd.csv', window_size=WINDOW_SIZE)

train_size = int(len(all_cyclones) * 0.7)
val_size = int(len(all_cyclones) * 0.85)

train_final = all_cyclones[:train_size]
val_final = all_cyclones[train_size:val_size]
test_final = all_cyclones[val_size:]
test_initials = all_initials[val_size:]

# 2. HUẤN LUYỆN
print(f'\n2. Quá trình Huấn luyện (Train: {len(train_final)}, Val: {len(val_final)})...')
scaler_X, scaler_Y, train_losses, val_losses = Training.training(
    data_train=train_final, data_val=val_final, window_size=WINDOW_SIZE, 
    first_hid=first_hid_cfg, second_hid=second_hid_cfg
)

print('-> Đang vẽ đồ thị hàm Loss...')
plt.figure(figsize=(10, 5))
plt.plot(train_losses, color='blue', linewidth=1.5, label='Training Loss')
plt.plot(val_losses, color='orange', linewidth=1.5, label='Validation Loss')
plt.title('Biểu đồ hàm Loss qua các Epochs (7 Đặc trưng)')
plt.xlabel('Epoch')
plt.ylabel('Giá trị Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('Training_Loss_Curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. KIỂM THỬ VÀ KHÔI PHỤC TỌA ĐỘ GỐC
print('\n3. Quá trình Kiểm thử...')
_, _, Y_true, Y_pred = Testing.testing(
    data_test=test_final, window_size=WINDOW_SIZE,
    first_hid=first_hid_cfg, second_hid=second_hid_cfg,
    scaler_X=scaler_X, scaler_Y=scaler_Y
)

for idx in range(len(Y_true)):
    start_lon, start_lat = test_initials[idx]
    Y_true[idx][:, 0] += start_lon
    Y_true[idx][:, 1] += start_lat
    Y_pred[idx][:, 0] += start_lon
    Y_pred[idx][:, 1] += start_lat

a_test_abs = np.concatenate(Y_pred, axis=0)
b_test_abs = np.concatenate(Y_true, axis=0)

# 4. ĐÁNH GIÁ (TRÊN TỌA ĐỘ TUYỆT ĐỐI)
mse_score = mean_squared_error(b_test_abs, a_test_abs)
all_point_distances = haversine_distance(b_test_abs, a_test_abs)
mean_km_error = np.mean(all_point_distances)

print(f'\nMSE: {mse_score:.5f} | Sai số trung bình (Haversine): {mean_km_error:.2f} km')

# --- BỔ SUNG: THỐNG KÊ & VẼ BIỂU ĐỒ PHÂN BỐ SAI SỐ TỪNG CƠN BÃO ---
print('\n--- BẢNG THỐNG KÊ PHÂN BỐ SAI SỐ ---')

# Khởi tạo bộ đếm cho các khoảng sai số
error_bins = {
    '< 40 km': 0, 
    '40 - 70 km': 0, 
    '70 - 90 km': 0, 
    '90 - 120 km': 0, 
    '120 - 150 km': 0, 
    '> 150 km': 0
}

# 1. Tính sai số trung bình của từng cơn bão và phân loại
for i in range(len(Y_true)):
    true_traj = Y_true[i].reshape(-1, 2)
    pred_traj = Y_pred[i].reshape(-1, 2)
    if len(true_traj) == 0 or len(pred_traj) == 0: 
        continue
    
    # Tính trung bình sai số của cơn bão thứ i
    storm_err = np.mean(haversine_distance(true_traj, pred_traj))
    
    # Phân loại vào nhóm
    if storm_err < 40:
        error_bins['< 40 km'] += 1
    elif storm_err < 70:
        error_bins['40 - 70 km'] += 1
    elif storm_err < 90:
        error_bins['70 - 90 km'] += 1
    elif storm_err < 120:
        error_bins['90 - 120 km'] += 1
    elif storm_err < 150:
        error_bins['120 - 150 km'] += 1
    else:
        error_bins['> 150 km'] += 1

# 2. In bảng thống kê ra màn hình console
df_stats = pd.DataFrame(list(error_bins.items()), columns=['Khoảng sai số', 'Số lượng cơn bão'])
print(df_stats.to_string(index=False))

# 3. Vẽ biểu đồ phân bố (Bar Chart) phong cách giống biểu đồ Loss
plt.figure(figsize=(10, 5))
labels = list(error_bins.keys())
counts = list(error_bins.values())

# Vẽ cột, dùng zorder=3 để cột đè lên lưới
bars = plt.bar(labels, counts, color='orange', edgecolor='blue', width=0.6, zorder=3)

# Ghi số lượng cụ thể lên đầu mỗi cột
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
             ha='center', va='bottom', fontweight='bold', color='black')

plt.title('Thống kê Số lượng Cơn bão theo Khoảng Sai số', fontsize=12)
plt.xlabel('Khoảng sai số trung bình (km)', fontsize=11)
plt.ylabel('Số lượng cơn bão', fontsize=11)

# Thêm lưới đứt đoạn giống ảnh mẫu của bạn
plt.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

# Lưu ảnh
output_chart = 'Error_Distribution_Chart.png'
plt.savefig(output_chart, dpi=150, bbox_inches='tight')
plt.close()
print(f"-> Đã xuất đồ thị thống kê sai số ra tệp '{output_chart}'")

# 5. VẼ ĐỒ THỊ (+12H)
output_dir = 'Cyclone_Plots_Custom'
os.makedirs(output_dir, exist_ok=True)

for idx in range(len(Y_true)):
    true_traj = Y_true[idx].reshape(-1, 2)
    pred_traj = Y_pred[idx].reshape(-1, 2)
    
    if len(true_traj) == 0 or len(pred_traj) == 0: continue

    start_lon, start_lat = test_initials[idx]
    history_rel = test_final[idx][:WINDOW_SIZE] 
    history_abs_x = history_rel[:, 0] + start_lon
    history_abs_y = history_rel[:, 1] + start_lat
    history_x = np.append(history_abs_x, true_traj[0, 0])
    history_y = np.append(history_abs_y, true_traj[0, 1])

    storm_km_dist = haversine_distance(true_traj, pred_traj)
    storm_avg_error = np.mean(storm_km_dist) if len(storm_km_dist) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_x, history_y, color='gray', linestyle='--', marker='.', alpha=0.7, label=f'Dữ liệu mồi ({WINDOW_SIZE} mốc)')
    ax.plot(true_traj[:, 0], true_traj[:, 1], 'k-o', markersize=3, linewidth=1.5, label='Thực tế')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--o', markersize=3, linewidth=1.5, label=f'Dự báo +12h (RMSE: {storm_avg_error:.2f} km)')
    ax.plot(start_lon, start_lat, 'go', markersize=8, label='Vị trí xuất phát (t0)')

    all_x = np.concatenate([history_abs_x, true_traj[:, 0], pred_traj[:, 0]])
    all_y = np.concatenate([history_abs_y, true_traj[:, 1], pred_traj[:, 1]])
    ax.set_xlim(np.min(all_x) - 3, np.max(all_x) + 3)
    ax.set_ylim(np.min(all_y) - 3, np.max(all_y) + 3)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel('Kinh độ - Longitude (°E)')
    ax.set_ylabel('Vĩ độ - Latitude (°N)')
    ax.set_title(f'DỰ BÁO QUỸ ĐẠO BÃO (+12H) - MẠNG 3 LỚP - ID: {idx}')
    ax.grid(True, which='major', linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize='small')

    plt.savefig(os.path.join(output_dir, f'Cyclone_{idx}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f'-> Đã xuất đồ thị địa lý tuyệt đối vào thư mục "{output_dir}/".')