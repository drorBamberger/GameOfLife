from functions import *

def generate_split_indexes_numbers(size,split_size,pixel):
    i_pixel,j_pixel = find_loc(pixel,size)
    #print(i_pixel,j_pixel)
    n = split_size-2
    result = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            #print(i,j)
            
            i_res = i+i_pixel
            if i_res <0:
                i_res += size
            if i_res >= size:
                i_res -= size
            
            j_res = j+j_pixel
            if j_res <0:
                j_res += size
            if j_res >=size:
                j_res -= size
            
            result.append(find_pixel(i_res, j_res,size))
    return result

def extract_selected_columns(size, row, selected_cols):
    num_cols = len(row)
    num_blocks = num_cols // (size*size)
    selected_data = []

    for block_idx in range(num_blocks):
        offset = block_idx * (size*size)
        for col in selected_cols:
            col_name = f'Col_{offset + col + 1}'
            selected_data.append(row[col_name])

    return pd.DataFrame([selected_data], columns=[f'Col_{i}' for i in range(len(selected_data))])

def print_boards_from_df_row(row, size):
    num_values = len(row)
    board_size = size * size
    num_boards = num_values // board_size

    for board_idx in range(num_boards):
        print(f"\nBoard {board_idx + 1}:")
        start = board_idx * board_size
        board_values = row[start:start + board_size]
        
        for i in range(size):  # שורות
            row_str = ""
            for j in range(size):  # עמודות
                val = board_values[i * size + j]
                row_str += "⬛" if val == 1 else "⬜"
            print(row_str)


"""def generate_split_indexes(size,n,i_pixel,j_pixel):
    result = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            i_res = i+i_pixel
            if i_res <=0:
                i_res+=size
            j_res = j+j_pixel
            if j_res <=0:
                j_res+=size
            result.append((i_res, j_res))
    return result"""
