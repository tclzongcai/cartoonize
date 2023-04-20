from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator
import io
import base64

app = Flask(__name__)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load('weights/paprika.pt', map_location='cpu'))
model.eval()

def load_image(image_bytes, x32=False):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cartoonize', methods=['POST'])
def cartoonize():
    # 获取上传的文件
    uploaded_file = request.files.get('file', '')
    file_extension = uploaded_file.filename.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        return 'Unsupported or missing file type', 400

    #获取其他参数
    upsample_align = bool(request.form.get('upsample_align'))
    x32 = bool(request.form.get('x32'))

    # 处理图像
    img = load_image(uploaded_file.read(), x32)

    with torch.no_grad():
        input_tensor = to_tensor(img).unsqueeze(0) * 2 - 1
        output_tensor = model(input_tensor.to(device), upsample_align).detach().cpu()
        output_tensor = output_tensor.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        output_img = to_pil_image(output_tensor)

    # 将处理的图像转换为数据 URL
    buff = io.BytesIO()
    output_img.save(buff, format="JPEG")
    encoded_img = base64.b64encode(buff.getvalue()).decode()

    return jsonify({'status': 'success', 'result_image': 'data:image/jpeg;base64,' + encoded_img})

if __name__ == '__main__':
    app.run(debug=True)
