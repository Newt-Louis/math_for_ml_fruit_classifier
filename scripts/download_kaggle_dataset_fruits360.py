import os
import subprocess
import sys

def download_dataset():
    script_dir = os.path.dirname(__file__)
    DATA_PATH = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw'))
    # Kiểm tra tồn tại trước khi thực hiện tải
    os.makedirs(DATA_PATH, exist_ok=True)

    print(f"Bắt đầu tải bộ dữ liệu 'moltean/fruits'...")

    # Xây dựng lệnh để chạy Kaggle CLI
    # -d: tên bộ dữ liệu
    # -p: đường dẫn để lưu file
    # --unzip: tự động giải nén sau khi tải xong
    command = [
        "kaggle", "datasets", "download",
        "-d", "moltean/fruits",
        "-p", DATA_PATH,
        "--unzip"
    ]

    try:
        print("Bắt đầu tải và giải nén bộ dữ liệu bằng Kaggle CLI...")
        # Chạy lệnh
        # check=True sẽ báo lỗi nếu lệnh chạy không thành công
        subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print("\n✅ Tải và giải nén thành công!")

    except FileNotFoundError:
        print("\n❌ Lỗi: Không tìm thấy lệnh 'kaggle'.")
        print("Vui lòng đảm bảo bạn đã cài đặt thư viện 'kaggle' và nó có trong PATH của hệ thống.")
        sys.exit(1)  # Thoát script với mã lỗi

    except subprocess.CalledProcessError as e:
        print("\n❌ Lỗi khi thực thi lệnh Kaggle CLI.")
        print(f"Lệnh trả về lỗi: {e.stderr}")
        print("Vui lòng kiểm tra lại thông tin xác thực (kaggle.json) và tên bộ dữ liệu.")
        sys.exit(1)  # Thoát script với mã lỗi

if __name__ == "__main__":
    download_dataset()