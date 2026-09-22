import numpy as np

def data_processing(table, window_size):
    table = np.array(table, dtype=np.float64)

    if table.ndim == 3:
        table = np.squeeze(table, axis=1)

    row_num = table.shape[0]
    col_num = table.shape[1] if table.ndim > 1 else 1

    if col_num == 0:
        col_num = 2

    X = []
    Y = []

    # Logic của bạn: Giới hạn vòng lặp trừ đi 3 để không bị IndexError
    for j in range(window_size, row_num - 3):
        window_tmp = np.zeros((window_size, col_num), dtype=np.float64)
        
        # Cửa sổ đầu vào: 4 điểm quá khứ
        for k in range(window_size):
            window_tmp[window_size - k - 1, :] = table[j - k - 1, :]

        # Nhãn dự báo: cách mốc hiện tại đúng 3 bước
        target_tmp = table[j + 3, :2].reshape(1, 2)

        X.append(window_tmp)
        Y.append(target_tmp)

    if len(X) == 0:
        return np.zeros((0, window_size, col_num), dtype=np.float64), np.zeros((0, 1, 2), dtype=np.float64)

    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)