import os
from typing import Annotated, Optional
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

# Ukuran maksimal untuk matriks piksel
MAX_MATRIX_SIZE = 20

def resize_for_matrix(img, max_size=MAX_MATRIX_SIZE):
    """Resize gambar untuk tampilan matriks agar tidak terlalu besar"""
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = min(max_size / h, max_size / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def create_histogram_chart(img, filename):
    """Membuat histogram RGB"""
    plt.figure(figsize=(8, 5))
    
    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    for i, (color, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=label)
    
    plt.title('Histogram RGB')
    plt.xlabel('Nilai Piksel (0-255)')
    plt.ylabel('Jumlah Piksel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    hist_filename = f"hist_{filename.split('.')[0]}.png"
    hist_path = os.path.join("static", "uploads", hist_filename)
    plt.savefig(hist_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return hist_filename


def normalize_to_uint8(image):
    """Normalize float or wide-range image arrays into uint8 for saving/display."""
    if image is None:
        return None

    if image.dtype == np.uint8:
        return image

    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def apply_convolution(image, kernel_type="average"):
    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        return image

    return cv2.filter2D(image, -1, kernel)


def apply_zero_padding(image, padding_size=20):
    return cv2.copyMakeBorder(
        image,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def apply_filter(image, filter_type="low"):
    if filter_type == "low":
        return cv2.GaussianBlur(image, (5, 5), 0)

    if filter_type == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    if filter_type == "band":
        low_pass = cv2.GaussianBlur(image, (9, 9), 0)
        high_pass = cv2.subtract(image, low_pass)
        return cv2.add(low_pass, high_pass)

    return image


def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return normalize_to_uint8(magnitude_spectrum)


def reduce_periodic_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    radius = 30
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return normalize_to_uint8(img_back)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: Annotated[UploadFile, File(...)]):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1].lower()
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    
    # Buat histogram
    hist_filename = create_histogram_chart(img, filename)
    
    # Resize gambar untuk tampilan matriks
    img_small = resize_for_matrix(img, MAX_MATRIX_SIZE)
    small_h, small_w = img_small.shape[:2]
    b_small, g_small, r_small = cv2.split(img_small)
    
    # Data matriks untuk tampilan
    matrix_data = []
    for y in range(small_h):
        row = []
        for x in range(small_w):
            row.append({
                'r': int(r_small[y][x]),
                'g': int(g_small[y][x]),
                'b': int(b_small[y][x])
            })
        matrix_data.append(row)

    return templates.TemplateResponse("display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "hist_path": f"/static/uploads/{hist_filename}",
        "matrix_data": matrix_data,
        "matrix_height": small_h,
        "matrix_width": small_w,
        "original_height": height,
        "original_width": width,
        "filename": file.filename
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    operation: Annotated[str, Form(...)],
    value: Annotated[int, Form(...)]
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: Annotated[UploadFile, File(...)],
    operation: Annotated[str, Form(...)],
    file2: Annotated[Optional[UploadFile], File()] = None,
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if img2 is None:
            return HTMLResponse("Gambar kedua tidak dapat dibaca.", status_code=400)

        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: Annotated[UploadFile, File(...)]):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: Annotated[UploadFile, File(...)]):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })

@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: Annotated[UploadFile, File(...)]):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: Annotated[UploadFile, File(...)], ref_file: Annotated[UploadFile, File(...)]):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)

    #jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR

    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: Annotated[UploadFile, File(...)]):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })


@app.get("/frequency/", response_class=HTMLResponse)
async def frequency_form(request: Request):
    return templates.TemplateResponse("frequency.html", {"request": request})


@app.post("/frequency/", response_class=HTMLResponse)
async def process_frequency(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    operation: Annotated[str, Form(...)],
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return HTMLResponse("Gambar tidak dapat dibaca", status_code=400)

    original_path = save_image(img, "freq_original")

    if operation == "conv_average":
        result_img = apply_convolution(img, "average")
    elif operation == "conv_sharpen":
        result_img = apply_convolution(img, "sharpen")
    elif operation == "conv_edge":
        result_img = apply_convolution(img, "edge")
    elif operation == "zero_padding":
        result_img = apply_zero_padding(img, 20)
    elif operation == "low_pass":
        result_img = apply_filter(img, "low")
    elif operation == "high_pass":
        result_img = apply_filter(img, "high")
    elif operation == "band_pass":
        result_img = apply_filter(img, "band")
    elif operation == "fourier":
        result_img = apply_fourier_transform(img)
    elif operation == "reduce_periodic_noise":
        result_img = reduce_periodic_noise(img)
    else:
        return HTMLResponse("Operasi tidak dikenali", status_code=400)

    result_img = normalize_to_uint8(result_img)
    result_path = save_image(result_img, operation)

    return templates.TemplateResponse(
        "frequency.html",
        {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": result_path,
            "selected_operation": operation,
        },
    )

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

